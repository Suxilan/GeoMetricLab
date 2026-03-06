# Code adapted from Lu-Feng/CricaVPR: https://github.com/Lu-Feng/CricaVPR

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AggregatorBase


class CricaVPR(AggregatorBase):
    """CricaVPR-style aggregator.

    Core logic migrated from official CricaVPR:
    1) L2-normalize patch features on channel dim.
    2) GeM pool multi-scale regions (2x2 and 3x3 non-overlapping windows).
    3) Concatenate with CLS token -> 14 tokens total.
    4) Transformer encoder over these tokens.
    5) Flatten and L2-normalize as the final global descriptor.

    Notes:
    - The original implementation uses ``batch_first=False`` and feeds tensor as
      (B, T, C), which effectively performs attention on the batch dimension.
      We preserve that behavior by default via ``batch_first=False``.
    - This aggregator requires ViT-style ``cls_token`` input.
    """

    def __init__(
        self,
        in_channels: int = None,
        p: float = 3.0,
        eps: float = 1e-6,
        nhead: int = 16,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = False,
        l2_before_gem: bool = True,
    ):
        super().__init__(in_channels=in_channels)
        if in_channels is None:
            raise ValueError("CricaVPR requires in_channels")
        if in_channels % nhead != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by nhead ({nhead})"
            )

        self.in_channels = in_channels
        self.out_channels = 14 * in_channels
        self.eps = float(eps)
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.l2_before_gem = bool(l2_before_gem)
        self.batch_first = bool(batch_first)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=self.batch_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _gem_pool_2d(self, x: torch.Tensor) -> torch.Tensor:
        """GeM over full spatial extent for each region tensor (B, C, h, w) -> (B, C)."""
        p = self.p.clamp_min(1.0)
        pooled = F.avg_pool2d(x.clamp_min(self.eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
        return pooled[..., 0, 0]

    @staticmethod
    def _split_bounds(length: int, num_splits: int) -> List[Tuple[int, int]]:
        """Split [0, length) into ``num_splits`` contiguous windows."""
        boundaries = torch.linspace(0, length, steps=num_splits + 1).round().to(torch.int64).tolist()
        bounds: List[Tuple[int, int]] = []
        for i in range(num_splits):
            s, e = int(boundaries[i]), int(boundaries[i + 1])
            if e <= s:
                e = min(length, s + 1)
            bounds.append((s, e))
        return bounds

    def _regional_tokens(self, x: torch.Tensor, grid: int) -> List[torch.Tensor]:
        """Extract GeM-pooled tokens from a grid x grid partition."""
        _, _, h, w = x.shape
        hs = self._split_bounds(h, grid)
        ws = self._split_bounds(w, grid)

        tokens: List[torch.Tensor] = []
        for h0, h1 in hs:
            for w0, w1 in ws:
                region = x[:, :, h0:h1, w0:w1]
                tokens.append(self._gem_pool_2d(region))
        return tokens

    def forward(
        self,
        x: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if cls_token is None:
            raise ValueError("CricaVPR requires cls_token from ViT backbone.")

        if x.dim() == 3:
            # (B, N, C) -> (B, C, H, W)
            b, n, c = x.shape
            side = int(math.sqrt(n))
            if side * side != n:
                raise ValueError(f"Patch token number must be a perfect square, got N={n}")
            x = x.transpose(1, 2).reshape(b, c, side, side)
        elif x.dim() != 4:
            raise ValueError(f"x must be 4D (B, C, H, W) or 3D (B, N, C), got {tuple(x.shape)}")

        if self.l2_before_gem:
            x = F.normalize(x, p=2, dim=1)

        # 1 (CLS) + 4 (2x2) + 9 (3x3) = 14 tokens
        region_tokens = self._regional_tokens(x, grid=2) + self._regional_tokens(x, grid=3)
        token_list = [cls_token] + region_tokens
        tokens = torch.stack(token_list, dim=1)  # (B, 14, C)

        encoded = self.encoder(tokens)  # preserve official behavior when batch_first=False
        desc = encoded.reshape(encoded.size(0), -1)

        return desc, None
