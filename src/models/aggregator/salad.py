"""
    Optimal Transport Aggregation for Visual Place Recognition

    Paper: https://arxiv.org/abs/2311.15937
    Code repo: https://github.com/serizba/salad

    Reference:
    @InProceedings{Izquierdo_CVPR_2024_SALAD,
        author    = {Izquierdo, Sergio and Civera, Javier},
        title     = {Optimal Transport Aggregation for Visual Place Recognition},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2024},
    }
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)

# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score = 1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n-m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(
        log_a,
        log_b,
        S_aug,
        num_iters=num_iters,
        reg=reg
    )
    return log_P - norm


from typing import Tuple, Optional, Any, Dict
from .base import AggregatorBase

# Code adapted from SALAD, GPL-3.0 license
# https://github.com/serizba/salad/blob/main/models/aggregators/salad.py
class Salad(AggregatorBase):
    """Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) Aggregator.
    
    Args:
        num_channels (int): Number of channels in the input feature map. Defaults to 1536.
        num_clusters (int): Number of clusters. Defaults to 64.
        cluster_dim (int): Dimension of cluster features. Defaults to 128.
        token_dim (int): Dimension of token features. Defaults to 256.
        dropout (float): Dropout rate. Defaults to 0.3. 
    """
    def __init__(
        self,
        num_channels: int = 1536,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.out_channels = num_clusters * cluster_dim + token_dim

        if dropout > 0:
            dropout_layer = nn.Dropout(dropout)
        else:
            dropout_layer = nn.Identity()

        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.))

    def forward(self, x: torch.Tensor, cls_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            cls_token (Optional[torch.Tensor]): CLS token of shape (B, C). Required for Salad.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: 
                - Global descriptor.
                - Dictionary containing assignment maps ('attn').
        """
        # x: (B, C, H, W)
        # cls_token: (B, C) optional
        B, C, H, W = x.shape

        if cls_token is None:
            # If no CLS token is provided (e.g. CNN), we can use GAP of x as a proxy or raise error.
            # Original Salad implementation expects a global token.
            # Here we will use GAP if cls_token is missing, to be compatible with CNNs.
            cls_token = x.mean(dim=(2, 3))
            
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(cls_token)

        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)  # (B, K+1, H*W)

        dustbin_prob = p[:, -1, :]  # (B, H*W)
        discard_map = dustbin_prob.view(B, 1, H, W)

        p = p[:, :-1, :]  # (B, K, H*W)
        assign_map = p.view(B, self.num_clusters, H, W)

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        f = torch.cat([
            F.normalize(t, p=2, dim=-1),
            F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1)
        ], dim=-1)
        
        return f, {
            "assign_map": assign_map,
            "discard_map": discard_map,
        }