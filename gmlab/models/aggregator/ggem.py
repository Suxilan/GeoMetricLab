"""Static grouped GeM, following the paper's Appendix B.1, Algorithm 1.

Reference: Ko et al., Group Generalized Mean Pooling for Vision Transformer.
https://arxiv.org/abs/2212.04114v1

This is an independent implementation of the published pooling equations.
Algorithm 1 groups channels by ``channel_index % groups`` after reshaping
tokens to [B,N,C/groups,groups]. This interleaved assignment differs from the
contiguous groups described in the main text; we follow the explicit algorithm.
Groups of final channels are not assumed to identify attention heads.
"""

import math

import torch
from torch import nn

from .base import AggregatorBase


class GGeM(AggregatorBase):
    """Pool [B,C,H,W] features with one static trainable exponent per group.

    Return descriptors [B,C] and auxiliary ``p`` [B,groups,1], expanded across
    images for consistent observation. The upstream feature adapter must remove
    any class token before constructing the feature map.

    ``mode="clamp"`` follows Algorithm 1. ``mode="abs"`` explicitly selects
    GeoMetricLab's magnitude-pooling variant for comparison with DGeM. Native
    autocast and dtype promotion are retained, with no forced FP32 computation
    or output cast. Exponents have no runtime clipping or reparameterization;
    callers must keep them away from zero, as for the shared GeM parameter.
    """

    def __init__(
        self,
        in_channels: int,
        groups: int = 12,
        p: float = 3.0,
        eps: float = 1e-6,
        mode: str = "clamp",
    ) -> None:
        super().__init__(in_channels=in_channels)
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if type(groups) is not int or groups < 1 or in_channels % groups:
            raise ValueError("groups must be a positive divisor of in_channels")
        if not math.isfinite(p) or p <= 0:
            raise ValueError("Initial p must be finite and positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        if mode not in {"clamp", "abs"}:
            raise ValueError("mode must be 'clamp' or 'abs'")
        self.out_channels = in_channels
        self.groups = groups
        self.eps = float(eps)
        self.mode = mode
        self.p = nn.Parameter(torch.full((groups,), float(p)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Aggregate spatial positions, preserving the original channel order."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("GGeM expects feature maps [B,in_channels,H,W]")
        if not x.is_floating_point() or x.shape[-2] * x.shape[-1] < 1:
            raise ValueError(
                "GGeM expects floating features with nonempty spatial axes"
            )

        tokens = x.flatten(2).transpose(1, 2)
        grouped = tokens.reshape(
            x.shape[0], tokens.shape[1], self.in_channels // self.groups, self.groups
        )
        values = (
            grouped.clamp_min(self.eps)
            if self.mode == "clamp"
            else grouped.abs() + self.eps
        )
        pooled = values.pow(self.p).mean(dim=1).pow(self.p.reciprocal())
        descriptor = pooled.reshape(x.shape[0], self.in_channels)
        return descriptor, {"p": self.p[None, :, None].expand(x.shape[0], -1, -1)}
