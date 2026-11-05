import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base import AggregatorBase


class DGeM(AggregatorBase):
    """Dynamic GeM (DGeM) Aggregator.

    Predicts a dynamic p for each sample using a small MLP on pooled features,
    then applies GeM with that p.

    Notes:
        - Keeps the original interface for compatibility.
        - Uses bounded p for stability.
        - GeM core is computed in fp32 by default.
        - `log_domain` is kept only for config compatibility and is not used.
    """

    def __init__(
        self,
        in_channels: int,
        p_min: float = 1.0,
        p_max: float = 8.0,
        eps: float = 1e-6,
        mode: str = "abs",
        hidden: int = 64,
        log_domain: bool = True,
    ):
        super().__init__(in_channels=in_channels)

        if p_max <= p_min:
            raise ValueError(f"p_max must be > p_min, got p_min={p_min}, p_max={p_max}")
        if mode not in {"clamp", "abs"}:
            raise ValueError(f"mode must be 'clamp' or 'abs', got {mode}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")

        self.out_channels = in_channels
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.eps = float(eps)
        self.mode = mode
        self.log_domain = bool(log_domain)  # compatibility only

        self.p_norm = nn.LayerNorm(in_channels)
        self.p_head = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self._init_p_head()

    def _init_p_head(self) -> None:
        # Initialize near standard GeM p=3.0, but respect bounds.
        target_p = min(max(3.0, self.p_min), self.p_max)
        y = max(target_p - self.p_min, 1e-6)
        # softplus^{-1}(y) = log(exp(y)-1)
        bias = math.log(math.exp(y) - 1.0)

        last_linear = self.p_head[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
            nn.init.constant_(last_linear.bias, bias)

    def _pow_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "clamp":
            return x.clamp_min(self.eps)
        if self.mode == "abs":
            return x.abs().add(self.eps)
        raise RuntimeError("Unreachable.")

    def _predict_p(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        stat = self._pow_input(x)
        s = F.adaptive_avg_pool2d(stat, (1, 1)).flatten(1)  # (B, C)
        s = self.p_norm(s)
        logits = self.p_head(s)  # (B, 1)
        p = self.p_min + F.softplus(logits)
        p = p.clamp_max(self.p_max)
        return p  # (B, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any | None]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - Global descriptor of shape (B, C).
                - Aux dict with predicted p (B, 1).
        """
        orig_dtype = x.dtype

        p = self._predict_p(x).view(-1, 1, 1, 1).float()  # (B, 1, 1, 1)
        base = self._pow_input(x).float()

        x_pow = base.pow(p)
        x_pow = F.avg_pool2d(x_pow, kernel_size=(x_pow.size(-2), x_pow.size(-1)))
        x_pow = x_pow.pow(p.reciprocal())

        y = x_pow.flatten(1).to(orig_dtype)
        aux = {"p": p.view(-1, 1, 1)}
        return y, aux
