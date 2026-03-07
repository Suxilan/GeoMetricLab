"""
    SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition

    Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b135d408253205ba501d55c6539bfc7-Abstract-Conference.html
    Code repo: https://github.com/Lu-Feng/SuperVLAD

    Reference:
    @inproceedings{lu2024supervlad,
        title={SuperVLAD: Compact and Robust Image Descriptors for Visual Place Recognition},
        author={Lu, Feng and Zhang, Xinyao and Ye, Canming and Dong, Shuting and Zhang, Lijun and Lan, Xiangyuan and Yuan, Chun},
        booktitle={Advances in Neural Information Processing Systems},
        volume={37},
        pages={5789--5816},
        year={2024}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any, Dict

from .base import AggregatorBase
from ..modules import DPN, SoftP


# Code adapted from SuperVLAD, MIT license
# https://github.com/Lu-Feng/SuperVLAD/blob/main/model/supervlad_layer.py
class SuperVLAD(AggregatorBase):
    """SuperVLAD Aggregator (follow the official SuperVLAD repo logic).

    The "official" SuperVLAD implementation you pasted does **not** explicitly
    subtract centroids during aggregation. It computes:

        v_k = sum_n a_{k|n} * x_n

    where a_{k|n} is computed by a 1x1 conv whose weights are initialized with
    alpha * L2-normalized centroids (no bias). Ghost clusters are appended and
    removed after aggregation.
    """

    def __init__(
        self,
        in_channels: int,
        num_clusters: int = 64,
        ghost_clusters: int = 1,
        normalize_input: bool = True,
        use_softp: bool = False,
        softp_alpha: float = 1.0,
        softp_hidden: int = 32,
        use_dpn: bool = False,
        dpn_d: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.ghost_clusters = ghost_clusters
        self.total_clusters = num_clusters + ghost_clusters
        self.normalize_input = normalize_input
        self.use_softp = use_softp
        self.use_dpn = use_dpn

        # Output excludes ghost clusters
        self.out_channels = num_clusters * in_channels

        # Official SuperVLAD uses bias=False
        self.assign = nn.Conv2d(in_channels, self.total_clusters, kernel_size=1, bias=False)
        
        if self.use_softp:
            self.softp = SoftP(alpha=softp_alpha, hidden=softp_hidden)
            
        if self.use_dpn:
            self.dpn = DPN(D=in_channels, d=dpn_d)

    def init_from_clusters(self, clsts: np.ndarray, traindescs: np.ndarray) -> None:
        """Initialize conv weights like the official SuperVLAD.

        Args:
            clsts: (K, C) kmeans centroids, with K == num_clusters.
            traindescs: (M, C) descriptors (recommended L2-normalized).
        """
        c_norm = np.linalg.norm(clsts, axis=1, keepdims=True)
        c_norm = np.maximum(c_norm, 1e-12)
        centroids_assign_real = clsts / c_norm

        dots = np.dot(centroids_assign_real, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]  # descending

        denom = float(np.mean(dots[0, :] - dots[1, :]))

        self.alpha = float((-np.log(0.01) / denom).item())
        print(f"[*] SuperVLAD initialized with alpha: {self.alpha:.4f}")

        with torch.no_grad():
            device = self.assign.weight.device
            dtype = self.assign.weight.dtype

            w = torch.zeros((self.total_clusters, self.in_channels, 1, 1), device=device, dtype=dtype)
            w_real = torch.from_numpy(self.alpha * centroids_assign_real).to(device=device, dtype=dtype)
            w[: self.num_clusters, :, 0, 0] = w_real
            self.assign.weight.copy_(w)

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

        B, C, H, W = x.shape

        if self.use_softp:
             # (B, C, H, W) -> (B, L, C)
            x = x.flatten(2).transpose(1, 2)
            x = self.softp(x)
            # (B, L, C) -> (B, C, H, W)
            x = x.transpose(1, 2).view(B, C, H, W)
        elif self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # (B, K+G, H, W)
        a = self.assign(x)
        a = F.softmax(a, dim=1)
        
        discard_map = a[:, self.num_clusters :, :, :].sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Flatten spatial dims
        x_flat = x.flatten(2)          # (B, C, N)
        a_flat = a.flatten(2)          # (B, K+G, N)

        # official aggregation (no centroid subtraction)
        v = torch.bmm(a_flat, x_flat.transpose(1, 2))  # (B, K, C)

        # Split real vs ghost for output + visualization
        a = a[:, : self.num_clusters, :, :]
        v = v[:, : self.num_clusters, :]
        
        if self.use_dpn:
            v = self.dpn(v)

        v = F.normalize(v, p=2, dim=2)
        v = v.flatten(1)

        return v, {
            "assign_map": a,
            "discard_map": discard_map,
        }