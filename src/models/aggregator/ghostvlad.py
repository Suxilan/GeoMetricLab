import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any, Dict, Optional

from .base import AggregatorBase


class GhostVLAD(AggregatorBase):
    """GhostVLAD Aggregator.

    A NetVLAD-style residual aggregator with **ghost clusters**.

    - Uses soft assignment over (K + G) clusters.
    - Aggregates residuals per cluster.
    - Drops the last G (ghost) clusters from the final descriptor.

    Notes:
        This implementation follows the project conventions:
        - Accepts either (B, C, H, W) or ViT patch tokens (B, N, C).
        - Returns (descriptor, aux_info).
        - Leaves final L2 normalization to GeoEncoder (via bn + final_norm).
    """

    def __init__(
        self,
        in_channels: int,
        num_clusters: int = 64,
        ghost_clusters: int = 1,
        normalize_input: bool = True,  # close for DINO backbones
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.ghost_clusters = ghost_clusters
        self.total_clusters = num_clusters + ghost_clusters
        self.normalize_input = normalize_input

        # Output excludes ghost clusters
        self.out_channels = num_clusters * in_channels

        # NetVLAD-style distance logits (supports KMeans init)
        self.assign = nn.Conv2d(in_channels, self.total_clusters, kernel_size=1, bias=True)

        # Cluster centers used for residuals (including ghost)
        self.centers = nn.Parameter(torch.rand(self.total_clusters, in_channels))

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
        print(f"[*] GhostVLAD initialized with alpha: {self.alpha:.4f}")

        with torch.no_grad():
            device = self.centers.device
            dtype = self.centers.dtype
            
            # centers: real clusters from KMeans, ghost clusters random
            self.centers[:self.num_clusters].copy_(torch.from_numpy(clsts_f32).to(device=device, dtype=dtype))

            # Initialize assignment logits like NetVLAD for the real clusters.
            # For ghost clusters, start neutral (0) and let gate_beta / learning shape them.
            w_real = (2.0 * self.alpha * torch.from_numpy(clsts_f32)).unsqueeze(-1).unsqueeze(-1)
            self.assign.weight[:self.num_clusters].copy_(w_real.to(device=device, dtype=self.assign.weight.dtype))

            centers_sq_norm = torch.from_numpy(np.sum(clsts_f32 * clsts_f32, axis=1))
            centers_sq_norm = centers_sq_norm.to(device=device, dtype=self.assign.bias.dtype)
            b_real = (-self.alpha * centers_sq_norm)
            self.assign.bias[:self.num_clusters].copy_(b_real)

            if self.ghost_clusters > 0:
                nn.init.xavier_uniform_(self.centers[self.num_clusters:])
                nn.init.xavier_uniform_(self.assign.weight[self.num_clusters:])
                nn.init.zeros_(self.assign.bias[self.num_clusters:])

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward.

        Args:
            x: (B, C, H, W) or (B, N, C) patch tokens.

        Returns:
            (B, K*C) descriptor and aux dict.
        """
        if x.dim() == 3:
            # (B, N, C) -> (B, C, N, 1)
            x = x.transpose(1, 2).unsqueeze(-1)
        elif x.dim() != 4:
            raise ValueError(f"Expected x with 3 or 4 dims, got shape {tuple(x.shape)}")

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # Soft assignment
        a = self.assign(x)
        a = F.softmax(a, dim=1)

        discard_map = a[:, self.num_clusters :, :, :].sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Flatten spatial dims
        x_flat = x.flatten(2)  # (B, C, N)
        a_flat = a.flatten(2)  # (B, K+G, N)

        # Standard NetVLAD residual aggregation over all (K+G) clusters
        v_x = torch.bmm(a_flat, x_flat.transpose(1, 2))  # (B, K+G, C)
        a_sum = a_flat.sum(dim=2)  # (B, K+G)
        v_c = a_sum.unsqueeze(2) * self.centers.unsqueeze(0)  # (B, K+G, C)
        v = v_x - v_c

        # Split real vs ghost for output + visualization
        a = a[:, : self.num_clusters, :, :]
        v = v[:, : self.num_clusters, :]

        # Intra-normalization (per cluster)
        v = F.normalize(v, p=2, dim=2)
        v = v.flatten(1)  # (B, K*C)

        return v, {
            "assign_map": a,
            "discard_map": discard_map,
        }

