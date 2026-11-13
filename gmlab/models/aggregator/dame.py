"""DAME pooling adapted from the authors' DAME_var implementation.

Source: scape-research/DAME-WEB, commit
 aefe876c80f36fad02221e98c027afcadb77e230, lib/TYY_DAME_model.py.
Code author: Tsun-Yi Yang. Licensed under GPL-3.0; see LICENSES/DAME-WEB.txt.
Paper: Yang et al., DAME WEB, ICCV Workshops 2019.
https://github.com/scape-research/DAME-WEB/blob/aefe876c80f36fad02221e98c027afcadb77e230/lib/TYY_DAME_model.py

Changes: GeoMetricLab adapter, argument/shape validation and modern no-grad
initialization. The pooling math, initialization and native AMP behavior follow
upstream. Backbone freezing, normalization and WEB remain pipeline concerns.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .base import AggregatorBase


class DAME(AggregatorBase):
    """Predict one exponent per image from raw spatial sample variances.

    Input: floating [B,C,H,W], with H*W > 1. Output: descriptor [B,C] and
    differentiable auxiliary ``p`` [B,1,1]. The upstream predictor uses
    sample variance (N-1 divisor), a C->1 affine layer, then
    ``p = 1 + delta_scale * sigmoid(alpha * fc(variance))``.

    Defaults match the official ResNet wrapper: delta_scale=4, ini_p=3, alpha=1.
    Upstream initializes weights to zero and bias to (ini_p-1)/delta_scale;
    consequently ini_p=3 yields p approximately 3.489837, NOT 3.
    Pooling clamps raw features to 1e-6 before exponentiation, without abs.
    No precision override or output cast is added; native autocast rules apply.
    ``fc`` parameter names allow loading the upstream DAME_var state dict.
    """

    def __init__(
        self,
        in_channels: int,
        delta_scale: float = 4.0,
        ini_p: float = 3.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(in_channels=in_channels)
        if not all(math.isfinite(v) for v in (delta_scale, ini_p, alpha)):
            raise ValueError("DAME parameters must be finite")
        if delta_scale <= 0 or alpha <= 0:
            raise ValueError("delta_scale and alpha must be positive")
        self.out_channels = in_channels
        self.delta_scale = float(delta_scale)
        self.ini_p = float(ini_p)
        self.alpha = float(alpha)
        self.eps = 1e-6
        self.fc = nn.Linear(in_channels, 1)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.fill_((self.ini_p - 1) / self.delta_scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return upstream descriptors and p through the aggregator contract."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("DAME expects feature maps [B,in_channels,H,W]")
        if not x.is_floating_point() or x.shape[-2] * x.shape[-1] <= 1:
            raise ValueError(
                "DAME sample variance requires floating features and H*W > 1"
            )
        variance = x.flatten(2).var(dim=-1, correction=1)
        logits = self.fc(variance).mul(self.alpha)
        p = 1 + logits.sigmoid() * self.delta_scale
        powered = x.clamp(min=self.eps).pow(p.unsqueeze(-1).unsqueeze(-1))
        pooled = F.avg_pool2d(powered, (x.shape[-2], x.shape[-1])).flatten(1)
        descriptor = pooled.pow(1.0 / p)
        return descriptor, {"p": p.unsqueeze(-1)}
