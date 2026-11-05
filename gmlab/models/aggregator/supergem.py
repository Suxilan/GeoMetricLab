"""Soft cluster assignment and dynamic GeM, preserving checkpoint and gradient math."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AggregatorBase


class DynamicPHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        p_min: float,
        p_max: float,
        target_p: float = 3.0,
    ):
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        self.norm = nn.LayerNorm(in_channels)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self._init_for_target_p(target_p)

    def _init_for_target_p(self, target_p: float) -> None:
        target_p = float(min(max(target_p, self.p_min), self.p_max))
        ratio = (target_p - self.p_min) / max(self.p_max - self.p_min, 1e-6)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        bias = math.log(ratio / (1.0 - ratio))  # logit

        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=1e-3)
            nn.init.constant_(last.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.net(x)
        p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(logits)
        return p


class SuperGeM(AggregatorBase):
    """Pool maps [B,C,H,W] or tokens [B,N,C] into [B,K*C].

    Assignment is detached only for predicting the per-cluster power statistics.
    Weighted power pooling keeps assignment gradients and uses FP32 intermediates,
    eight clusters per temporary chunk, then per-cluster L2 normalization. Return
    auxiliary assign_map [B,K,H,W] and p [B,1,K]. State keys match the old module.
    """

    def __init__(
        self,
        in_channels: int,
        num_clusters: int = 4,
        eps: float = 1e-6,
        use_dgem: bool = True,
        d_mode: str = "cluster",
        p_min: float = 1.0,
        p_max: float = 8.0,
        p_hidden: int | None = None,
        target_p: float = 3.0,
    ):
        super().__init__(in_channels=in_channels)

        if (
            type(in_channels) is not int
            or in_channels < 1
            or type(num_clusters) is not int
            or num_clusters < 1
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise ValueError("SuperGeM channels, clusters and eps must be positive")
        if (
            not all(math.isfinite(value) for value in (p_min, p_max, target_p))
            or p_min <= 0
            or p_max <= p_min
        ):
            raise ValueError(f"p_max must be > p_min, got p_min={p_min}, p_max={p_max}")

        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.eps = float(eps)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        if not use_dgem or d_mode != "cluster":
            raise ValueError(
                "SuperGeM implements only dynamic cluster-wise power pooling"
            )
        if p_hidden is not None and (type(p_hidden) is not int or p_hidden < 1):
            raise ValueError("p_hidden must be a positive integer")
        self.p_hidden = (
            p_hidden if p_hidden is not None else max(1, self.in_channels // 4)
        )
        self.out_channels = self.num_clusters * self.in_channels

        self.assign = nn.Conv2d(
            self.in_channels, self.num_clusters, kernel_size=1, bias=True
        )
        self.p_head = DynamicPHead(
            in_channels=self.in_channels,
            hidden=self.p_hidden,
            p_min=self.p_min,
            p_max=self.p_max,
            target_p=target_p,
        )

    def _pow_input(self, x: torch.Tensor) -> torch.Tensor:
        """Map signed backbone features to the legacy positive power domain."""
        return x.abs().add(self.eps)

    def _predict_p(
        self,
        base: torch.Tensor,
        a_flat: torch.Tensor,
    ) -> torch.Tensor:
        """Predict p [B,K,1] from base [B,N,C] and assignment [B,K,N].

        Only assignment in these statistics is detached: p prediction cannot
        change assignments through cluster-summary weights. Base still carries
        gradients, and the original assignments train through weighted pooling.
        """
        assignment = a_flat.detach()
        cluster_mass = assignment.sum(dim=2, keepdim=True).clamp_min(self.eps)
        summary = torch.bmm(assignment, base) / cluster_mass
        return self.p_head(summary)

    def _weighted_gem(
        self,
        base: torch.Tensor,
        a_flat: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Pool base [B,N,C] with assignment [B,K,N] and powers [B,K,1].

        Return [B,K,C] in FP32. Keep gradients through base, assignment and p;
        unlike the statistics path, assignment is deliberately not detached.
        At most eight clusters materialize [B,kc,N,C] powers at one time.
        """
        bsz, num_clusters, _ = a_flat.shape
        dim = base.size(-1)

        w_sum = a_flat.sum(dim=2, keepdim=True).clamp_min(self.eps)
        out = base.new_empty(bsz, num_clusters, dim)

        chunk = 8  # simple memory guard for K-wise dynamic p
        for k0 in range(0, num_clusters, chunk):
            k1 = min(k0 + chunk, num_clusters)

            a_chunk = a_flat[:, k0:k1, :].unsqueeze(-1)
            p_chunk = p[:, k0:k1, :].unsqueeze(-1)

            x_pow = base.unsqueeze(1).pow(p_chunk)
            num = (a_chunk * x_pow).sum(dim=2)

            out[:, k0:k1, :] = (num / w_sum[:, k0:k1, :]).pow(1.0 / p[:, k0:k1, :])

        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        x:
            - (B, C, H, W), or
            - (B, N, C)   -> treated as (B, C, N, 1)

        returns:
            v:   (B, K*C)
            aux: {"assign_map": (B, K, H, W), "p": (B, 1, K)}
        """
        orig_dtype = x.dtype

        if x.dim() == 3:
            x = x.transpose(1, 2).unsqueeze(-1)

        if x.dim() != 4:
            raise ValueError(f"x must be 4D (or 3D), got {tuple(x.shape)}")

        bsz, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"dim mismatch: got {channels}, expected {self.in_channels}"
            )

        a = F.softmax(self.assign(x), dim=1)
        a_flat = a.flatten(2).float()

        x_tokens = x.flatten(2).transpose(1, 2)
        base = self._pow_input(x_tokens).float()

        p = self._predict_p(base, a_flat).float()

        v = self._weighted_gem(base, a_flat, p)

        v = F.normalize(v, p=2, dim=2).flatten(1).to(orig_dtype)

        aux = {
            "assign_map": a,
            "p": p.transpose(1, 2),  # (B, 1, K)
        }
        return v, aux
