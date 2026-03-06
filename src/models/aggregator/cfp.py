import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, Optional, Dict
from ..modules import DPN
from .base import AggregatorBase

class MLP2(nn.Module):
    """Two-layer MLP: Linear -> Act -> Linear (for both FC and FP)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, in_dim]
        return self.net(x)

class CFP(AggregatorBase):
    """CFP Aggregator (EMVP core) wrapped for GeoMetricLab.

    Input:
      - (B, N, C) patch tokens, OR
      - (B, C, H, W) feature map (flattened into tokens)

    Output:
      - (B, D*K) descriptor, and aux dict.

    Note: GeoEncoder will still apply BN and (optionally) final L2 norm.
    """

    def __init__(
        self,
        in_channels: int,
        D: int = 128,
        K: int = 64,
        mlp_hidden: int = 512,
        cn_type: str = "softmax",  # "softmax" or "l2"
        use_dpnc: bool = True,
        dpn_d: int = 32,
        eps: float = 1e-6
    ):
        super().__init__(in_channels=in_channels)
        self.D = int(D)
        self.K = int(K)
        self.cn_type = str(cn_type).lower()
        self.use_dpnc = bool(use_dpnc)
        self.eps = float(eps)

        self.Fc = MLP2(in_channels, mlp_hidden, self.D, dropout=0.0)
        self.Fp = MLP2(in_channels, mlp_hidden, self.K, dropout=0.0)

        if self.use_dpnc:
            self.dpnc = DPN(D=self.D, d=dpn_d, eps=self.eps)

        self.out_channels = self.D * self.K

    def constant_normalize(self, p_logits: torch.Tensor) -> torch.Tensor:
        if self.cn_type == "softmax":
            return F.softmax(p_logits, dim=-1)
        elif self.cn_type == "l2":
            return F.normalize(p_logits, p=2, dim=-1)
        else:
            raise ValueError(f"Unsupported cn_type: {self.cn_type}")

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:

        if x.dim() == 4:
            # (B, C, H, W) -> (B, L, C)
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)
        elif x.dim() == 3:
            x_flat = x
        else:
            raise ValueError(f"CFP expects x with 3 or 4 dims, got shape {tuple(x.shape)}")

        # x_flat: [B, L, in_channels]
        C = self.Fc(x_flat)  # [B, L, D]
        P_logits = self.Fp(x_flat)  # [B, L, K]
        P = self.constant_normalize(P_logits)  # [B, L, K]

        # G = Fc(X)^T @ Fp(X) -> [B, D, K]
        G = torch.matmul(C.transpose(1, 2), P)

        if self.use_dpnc:
            # Treat K as L for DPN: [B, K, D]
            Gk = G.transpose(1, 2)
            Gk = self.dpnc(Gk)
            G = Gk.transpose(1, 2)

        g = G.reshape(G.shape[0], -1)  # [B, D*K]

        return g, {
            "assign_map": P.transpose(1, 2).view(B, self.K, H, W)  # [B, K, H, W] 
        }
