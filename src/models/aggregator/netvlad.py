import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple, Optional, Any, Dict
from .base import AggregatorBase
from ..modules import DPN, SoftP

class NetVLAD(AggregatorBase):
    """NetVLAD Aggregator.
    
    Aggregates local features into global descriptors using NetVLAD.
    
    Args:
        in_channels (int): Input feature dimension.
        num_clusters (int): Number of clusters (K). Defaults to 64.
        normalize_input (bool): Whether to L2 normalize input features. Defaults to True.
    """
    def __init__(
        self, 
        in_channels: int,
        num_clusters: int = 64,
        normalize_input: bool = True,  # close for DINO backbones
        use_softp: bool = False,
        softp_alpha: float = 1.0,
        softp_hidden: int = 32,
        use_dpn: bool = False,
        dpn_d: int = 32,
        aux_loss_weight: float = 0.0,
        aux_loss_type: str = "balance",
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.in_channels = in_channels
        self.normalize_input = normalize_input
        self.use_softp = use_softp
        self.use_dpn = use_dpn
        self.aux_loss_weight = float(aux_loss_weight)
        self.aux_loss_type = str(aux_loss_type).lower()
        self.out_channels = num_clusters * in_channels

        # Soft assignment
        self.assign = nn.Conv2d(in_channels, num_clusters, kernel_size=1, bias=True)

        # Cluster centers
        self.centers = nn.Parameter(torch.rand(num_clusters, in_channels))
        
        if self.use_softp:
            self.softp = SoftP(alpha=softp_alpha, hidden=softp_hidden)
            
        if self.use_dpn:
            self.dpn = DPN(D=in_channels, d=dpn_d)

        # K-means init placeholders
        self.clsts = None
        self.traindescs = None

        self._init_params()

    def _init_params(self) -> None:
        nn.init.xavier_uniform_(self.centers)

    def init_from_clusters(self, clsts: np.ndarray, traindescs: np.ndarray) -> None:
        """Initialize parameters from K-means clustering results (v2 with faiss KNN).
        
        Args:
            clsts (np.ndarray): Cluster centers of shape (K, C).
            traindescs (np.ndarray): Training descriptors of shape (M, C).
        """
        clsts_f32 = clsts.astype(np.float32, copy=False)
        c2 = np.sum(clsts_f32 * clsts_f32, axis=1, keepdims=True)  # (K,1)
        d2 = c2 + c2.T - 2.0 * (clsts_f32 @ clsts_f32.T)          # (K,K)
        np.fill_diagonal(d2, np.inf)
        nn_d2 = np.min(d2, axis=1)  # (K,)
        mean_nn = float(np.mean(nn_d2))

        self.alpha = float((-np.log(0.01) / (mean_nn + 1e-12)).item())

        print(f"[*] NetVLAD initialized with alpha: {self.alpha:.4f}")
        
        with torch.no_grad():
            device = self.centers.device
            dtype = self.centers.dtype
            
            # centers
            self.centers.copy_(torch.from_numpy(clsts_f32).to(device=device, dtype=dtype))
            
            # weight = 2 * alpha * centers [K, C, 1, 1]
            w = (2.0 * self.alpha * torch.from_numpy(clsts_f32)).unsqueeze(-1).unsqueeze(-1)
            self.assign.weight.copy_(
                w.to(device=device, dtype=self.assign.weight.dtype)
            )
            
            # bias = -alpha * ||centers||^2  [K]
            centers_sq_norm = torch.from_numpy(np.sum(clsts_f32 * clsts_f32, axis=1))
            centers_sq_norm = centers_sq_norm.to(device=device, dtype=self.assign.bias.dtype)
            self.assign.bias.copy_(-self.alpha * centers_sq_norm)

    def compute_aux_loss(self, assign_map: torch.Tensor) -> torch.Tensor:
        """Compute unsupervised anti-collapse auxiliary loss from assignment map.

        Args:
            assign_map: soft assignment map of shape (B, K, H, W) or (B, K, N)
        """
        if self.aux_loss_weight <= 0.0:
            return assign_map.new_zeros(())

        if assign_map.dim() == 4:
            local_sum = assign_map.sum(dim=(0, 2, 3))  # [K]
            local_count = float(assign_map.shape[0] * assign_map.shape[2] * assign_map.shape[3])
        elif assign_map.dim() == 3:
            local_sum = assign_map.sum(dim=(0, 2))
            local_count = float(assign_map.shape[0] * assign_map.shape[2])
        else:
            raise ValueError(f"assign_map must be 3D/4D, got shape: {tuple(assign_map.shape)}")

        p_local = local_sum / (local_count + 1e-12)

        # DDP global usage distribution (all_reduce), while preserving local gradients.
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            with torch.no_grad():
                global_sum = local_sum.detach().clone()
                global_count = torch.tensor(local_count, device=assign_map.device, dtype=assign_map.dtype)
                dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
                p_global = global_sum / (global_count + 1e-12)
            p = p_local + (p_global - p_local).detach()
        else:
            p = p_local

        p = p / (p.sum() + 1e-12)
        k = p.numel()
        uniform = torch.full_like(p, 1.0 / float(k))

        # KL(p || U): zero when balanced
        kl_loss = (p * (torch.log(p + 1e-12) - torch.log(uniform + 1e-12))).sum()
        # Simple balance loss
        balance_loss = ((p - uniform) ** 2).mean()

        raw = balance_loss if self.aux_loss_type != "kl" else kl_loss
        return self.aux_loss_weight * raw

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (B, N, C).

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: 
                - Global descriptor of shape (B, K*C).
                - Dictionary containing assignment maps ('attn').
        """
        # Handle input shape
        if x.dim() == 3:
            # (B, N, C) -> (B, C, N) -> (B, C, H, W) where H*W=N
            # For simplicity, treat as (B, C, N, 1)
            x = x.transpose(1, 2).unsqueeze(-1)
        
        B, C = x.shape[:2]
        H, W = x.shape[2], x.shape[3]
            
        if self.use_softp:
            # (B, C, H, W) -> (B, L, C)
            x = x.flatten(2).transpose(1, 2)
            x = self.softp(x)
            # (B, L, C) -> (B, C, H, W)
            x = x.transpose(1, 2).view(B, C, H, W)
        elif self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # L2 normalize across descriptor dim

        # Assignment a_{k|n}
        # (B, K, H, W)
        a = self.assign(x)                     
        a = F.softmax(a, dim=1)

        # Flatten spatial dims
        # x: (B, C, N)
        x_flat = x.flatten(2)
        # a: (B, K, N)
        a_flat = a.flatten(2)

        # Aggregation: v_kc = sum_n a_kn * (x_cn - c_kc)
        # First term: sum_n a_kn * x_cn
        # (B, K, N) @ (B, N, C) -> (B, K, C)
        # Note: x_flat is (B, C, N), so we transpose it
        v_x = torch.bmm(a_flat, x_flat.transpose(1, 2))
        
        # Second term: sum_n a_kn * c_kc = (sum_n a_kn) * c_kc
        # (B, K)
        a_sum = a_flat.sum(dim=2)
        # (B, K, C)
        v_c = a_sum.unsqueeze(2) * self.centers.unsqueeze(0)
        
        # Residuals
        v = v_x - v_c
        
        if self.use_dpn:
            v = self.dpn(v) # (B, K, C)
        
        # Intra-normalization (L2 per cluster)
        v = F.normalize(v, p=2, dim=2)
        
        # Flatten: (B, K*C)
        v = v.flatten(1)

        return v, {"assign_map": a}
