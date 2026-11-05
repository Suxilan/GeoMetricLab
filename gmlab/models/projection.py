from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ProjectionHead(nn.Module):
    """Simple 2-layer projection head (SimCLR style).

    Applies Linear -> BatchNorm -> ReLU -> Linear -> L2 normalization. Leading
    axes of [...,Din] are flattened into the BN sample axis, then restored in
    [...,Dout]. Use explicit masking before this head for padded input rows.
    """

    def __init__(self, in_dim: int, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.out_channels = out_dim
        hidden = hidden_dim
        out = out_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out, bias=False),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        if isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., D_in).

        Returns:
            z: Output tensor of shape (..., D_out).
        """
        if x.ndim < 2:
            raise ValueError("Projection inputs require a batch axis and feature axis")
        z = self.proj(x.reshape(-1, x.shape[-1])).reshape(
            *x.shape[:-1], self.out_channels
        )
        z = F.normalize(z, p=2, dim=-1)

        return z


class RetrievalHead(nn.Module):
    """Project [N,Din] to [N,Dout], then apply BatchNorm without final L2."""

    def __init__(self, in_dim=2048, out_dim=512):
        super().__init__()
        self.out_channels = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

        self.bn = nn.BatchNorm1d(out_dim)

        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x):
        feat = self.proj(x)
        feat = self.bn(feat)
        return feat


# MoCoProjection's MLP builder is adapted from MoCo-v3.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# The adapted builder uses CC BY-NC 4.0; see LICENSES/MoCo-v3.txt and NOTICE.
# https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
# Modified for GeoMetricLab on 2026-11-05: renamed builder arguments and
# integrated the builder with a standalone head and input validation.
class MoCoProjection(nn.Module):
    """
    MoCo/SimCLR style projection head: multi-layer MLP + optional last layer BN (no affine)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        nlayers: int = 3,
        last_bn: bool = False,
    ):
        super().__init__()
        if type(nlayers) is not int or nlayers < 1:
            raise ValueError("nlayers must be a positive integer")
        self.out_channels = out_dim
        self.head = self._build_mlp(
            nlayers, in_dim, hidden_dim, out_dim, last_bn=last_bn
        )

    def _build_mlp(self, nlayers, in_dim, hidden_dim, out_dim, last_bn=True):
        """Build native linear/normalization layers in execution order."""
        mlp = []
        for layer_index in range(nlayers):
            dim1 = in_dim if layer_index == 0 else hidden_dim
            dim2 = out_dim if layer_index == nlayers - 1 else hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if layer_index < nlayers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # SimCLR uses a BatchNorm without affine transformation
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def forward(
        self, x: torch.Tensor, node_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in]
            node_mask: [B, N]
        Returns:
            z: [B, N, D_out]
        """
        if x.ndim != 3:
            raise ValueError("Expected features [B,N,D]")
        if node_mask is None:
            return self.head(x.flatten(0, 1)).reshape(*x.shape[:2], self.out_channels)
        if (
            node_mask.dtype != torch.bool
            or node_mask.shape != x.shape[:2]
            or not node_mask.any()
        ):
            raise ValueError("node_mask must be nonempty boolean [B,N]")
        # Exclude padding before BN, including invalid/NaN values. Masked rows
        # are zero outputs and contribute neither statistics nor gradients.
        values = self.head(x[node_mask])
        output = values.new_zeros(*x.shape[:2], self.out_channels)
        output[node_mask] = values
        return output
