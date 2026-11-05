"""Structural Confidence Probe Pooling, preserving the existing SCPP objective."""

import torch
from torch import nn
from torch.nn import functional as F

from .base import AggregatorBase


class SCPP(AggregatorBase):
    """Pool feature maps [N,C,H,W] into normalized two-branch descriptors [N,2C].

    Both branches use learned confidence maps and fixed power exponents. The
    returned assign_map [N,1,H,W] is model auxiliary output, not an extra loss.
    Parameter names and operations match the original implementation so loaded
    SCPP weights retain their meaning.
    """

    def __init__(
        self, in_channels: int, p_s: float = 1.3, p_a: float = 4.6, eps: float = 1e-6
    ) -> None:
        super().__init__(in_channels)
        self.out_channels = in_channels * 2
        self.p_s, self.p_a, self.eps = float(p_s), float(p_a), float(eps)
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True,
        )
        self.confidence_proj_low = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.confidence_proj_high = nn.Conv2d(in_channels, 1, kernel_size=1)

    def _power_pool(
        self, features: torch.Tensor, confidence: torch.Tensor, exponent: float
    ) -> torch.Tensor:
        normalized = confidence / confidence.mean(dim=(2, 3), keepdim=True).clamp_min(
            self.eps
        )
        powered = features.abs().clamp_min(self.eps).pow(exponent)
        weighted_sum = (normalized * powered).sum(dim=(2, 3))
        weight_sum = normalized.sum(dim=(2, 3)).clamp_min(self.eps)
        return (weighted_sum / weight_sum).pow(1.0 / exponent)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict]:
        context = features + self.dwconv(features)
        support_confidence = self.confidence_proj_low(context).sigmoid()
        anchor_confidence = self.confidence_proj_high(context).sigmoid()
        support = self._power_pool(features, support_confidence, self.p_s)
        anchor = self._power_pool(features, anchor_confidence, self.p_a)
        descriptors = torch.cat(
            [F.normalize(support, dim=1), F.normalize(anchor, dim=1)], dim=1
        )
        return descriptors, {"assign_map": support_confidence}
