import torch
import torch.nn.functional as F
from torch import nn

from .base import AggregatorBase


class CGeM(AggregatorBase):
    """Channel-wise Dynamic GeM (simple version).

    - Predict one `p_c` per channel for each sample.
    """

    def __init__(
        self,
        in_channels: int,
        p: float = 3.0,
        mode: str = "abs",
        eps: float = 1e-6,
    ):
        super().__init__()
        if in_channels is None:
            raise ValueError("CGeM requires `in_channels`.")

        self.out_channels = in_channels
        self.in_channels = in_channels
        self.eps = float(eps)
        self.mode = str(mode).lower()

        self.p = nn.Parameter(torch.ones(1, in_channels, 1, 1) * p)

    def _pow_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "clamp":
            return x.clamp_min(self.eps)
        if self.mode == "abs":
            return x.abs().add(self.eps)
        raise ValueError(f"mode must be 'clamp' or 'abs', got {self.mode}")

    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        if x.dim() != 4:
            raise ValueError(
                f"CGeM expects 4D input (B,C,H,W), got shape {tuple(x.shape)}"
            )

        # Bound the gradient-scaled exponent to keep powers numerically stable.
        # p shape: (1, C, 1, 1)
        p = self.p.clamp(min=1.0, max=15.0)

        # Compute positive base energy before applying channel-wise powers.
        base = self._pow_input(x)
        y = F.adaptive_avg_pool2d(base.pow(p), (1, 1)).pow(1.0 / p).flatten(1)
        # if self.log_domain:
        #     x_pow = torch.exp(p * torch.log(base))
        # else:
        #     x_pow = base.pow(p)

        # Spatial pooling.
        # x_pool = F.adaptive_avg_pool2d(x_pow, (1, 1))

        # Invert the channel-wise power.
        # if self.log_domain:
        #     y_mag = torch.exp(torch.log(x_pool.clamp_min(self.eps)) / p)
        # else:
        #     y_mag = x_pool.pow(1.0 / p)

        # y_flat = y_mag.flatten(1)

        # Expose exponents for optional detached distribution monitoring.
        aux = {"p": p.view(-1).detach()}
        return y, aux
