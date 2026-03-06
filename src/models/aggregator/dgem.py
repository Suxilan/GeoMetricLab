import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any
import math
from .base import AggregatorBase


# class DGeM(AggregatorBase):
#     """Dynamic GeM (DGeM) Aggregator.

#     Predicts a dynamic p for each sample using a small MLP on pooled features,
#     then applies GeM with that p. The p is constrained to [p_min, p_max].
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         p_min: float = 1.0,
#         p_max: float = 10.0,
#         eps: float = 1e-6,
#         mode: str = "abs",
#         hidden: int = 64,
#         log_domain: bool = True,
#     ):
#         super().__init__(in_channels=in_channels)
#         self.out_channels = in_channels
#         self.p_min = float(p_min)
#         self.p_max = float(p_max)
#         self.eps = float(eps)
#         self.mode = mode
#         self.log_domain = bool(log_domain)

#         self.p_head = nn.Sequential(
#             nn.Linear(in_channels, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, 1),
#             nn.Sigmoid(),
#         )
#         self._init_p_head()

#     def _init_p_head(self) -> None:
#         # Initialize so sigmoid output maps to p=3.0 at start.
#         target_p = 3.0
#         p01 = (target_p - self.p_min) / (self.p_max - self.p_min)
#         p01 = float(max(min(p01, 1.0 - 1e-6), 1e-6))
#         bias = math.log(p01 / (1.0 - p01))
#         # Last Linear is index -2 in the Sequential
#         last_linear = self.p_head[-2]
#         if isinstance(last_linear, nn.Linear):
#             nn.init.zeros_(last_linear.weight)
#             nn.init.constant_(last_linear.bias, bias)

#     def _pow_input(self, x: torch.Tensor) -> torch.Tensor:
#         if self.mode == "clamp":
#             return x.clamp_min(self.eps)
#         if self.mode == "abs":
#             return x.abs().add(self.eps)
#         raise ValueError(f"mode must be 'clamp' or 'abs', got {self.mode}")

#     def _predict_p(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (B, C, H, W)
#         s = F.adaptive_avg_pool2d(x.abs(), (1, 1)).flatten(1)  # (B, C)
#         p01 = self.p_head(s)  # (B, 1) in [0,1]
#         p = self.p_min + (self.p_max - self.p_min) * p01
#         return p  # (B, 1)

#     def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
#         """Forward pass.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, C, H, W).

#         Returns:
#             Tuple[torch.Tensor, Dict[str, Any]]:
#                 - Global descriptor of shape (B, C).
#                 - Aux dict with predicted p (B, 1).
#         """
#         # x: (B, C, H, W)
#         p = self._predict_p(x).view(-1, 1, 1, 1)  # (B, 1, 1, 1)

#         base = self._pow_input(x)
#         if self.log_domain:
#             x_pow = torch.exp(p * torch.log(base))
#         else:
#             x_pow = base.pow(p)

#         x_pow = F.avg_pool2d(x_pow, (x_pow.size(-2), x_pow.size(-1)))
#         x_pow = x_pow.pow(1.0 / p)

#         y = x_pow.flatten(1)
#         # keep p for analysis: (B, 1, 1)
#         aux = {"p": p.view(-1, 1, 1)}
#         return y, aux


class DGeM(AggregatorBase):
    """Dynamic GeM (DGeM) Aggregator.

    Predicts a dynamic p for each sample using a small MLP on pooled features,
    then applies GeM with that p.
    """

    def __init__(
        self,
        in_channels: int,
        p_min: float = 1.0,
        eps: float = 1e-6,
        mode: str = "abs",
        hidden: int = 64,
        log_domain: bool = True,
    ):
        super().__init__(in_channels=in_channels)
        self.out_channels = in_channels
        self.p_min = float(p_min)
        self.eps = float(eps)
        self.mode = mode
        self.log_domain = bool(log_domain)

        self.p_head = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )
        self._init_p_head()

    def _init_p_head(self) -> None:
        # Initialize so p_min + softplus(z) == 3.0 at start.
        target_p = 3.0
        y = max(target_p - self.p_min, 1e-6)
        # softplus^{-1}(y) = log(exp(y) - 1)
        bias = math.log(math.exp(y) - 1.0)
        # Last Linear is index -2 in the Sequential
        last_linear = self.p_head[-2]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, bias)

    def _pow_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "clamp":
            return x.clamp_min(self.eps)
        if self.mode == "abs":
            return x.abs().add(self.eps)
        raise ValueError(f"mode must be 'clamp' or 'abs', got {self.mode}")

    def _predict_p(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        s = F.adaptive_avg_pool2d(x.abs(), (1, 1)).flatten(1)  # (B, C)
        p = self.p_min + self.p_head(s)  # (B, 1)
        return p  # (B, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - Global descriptor of shape (B, C).
                - Aux dict with predicted p (B, 1).
        """
        # x: (B, C, H, W)
        p = self._predict_p(x).view(-1, 1, 1, 1)  # (B, 1, 1, 1)

        base = self._pow_input(x)
        if self.log_domain:
            x_pow = torch.exp(p * torch.log(base))
        else:
            x_pow = base.pow(p)

        x_pow = F.avg_pool2d(x_pow, (x_pow.size(-2), x_pow.size(-1)))
        x_pow = x_pow.pow(1.0 / p)

        y = x_pow.flatten(1)
        # keep p for analysis: (B, 1, 1)
        aux = {"p": p.view(-1, 1, 1)}
        return y, aux