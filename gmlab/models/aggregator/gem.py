"""
Fine-tuning CNN Image Retrieval with No Human Annotation

Paper: https://arxiv.org/abs/1711.02512
Code repo: https://cmp.felk.cvut.cz/cnnimageretrieval/

Reference:
@article{radenovic2018fine,
    title={Fine-tuning CNN image retrieval with no human annotation},
    author={Radenovi{\'c}, Filip and Tolias, Giorgos and Chum, Ond{\v{r}}ej},
    journal={IEEE transactions on pattern analysis and machine intelligence},
    volume={41},
    number={7},
    pages={1655--1668},
    year={2018},
    publisher={IEEE}
}

"""

import torch
from torch import nn
from torch.nn import functional as F

from .base import AggregatorBase


# Adapted from cnnimageretrieval-pytorch (MIT).
# Copyright (c) 2016-2019 VRG, CTU Prague
# See LICENSES/GeM.txt and NOTICE for the retained license.
# https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
class GeM(AggregatorBase):
    """Generalized Mean Pooling (GeM) Aggregator.

    Paper: Fine-tuning CNN Image Retrieval with No Human Annotation (TPAMI 2018)

    Args:
        p (float): Initial power value. Defaults to 3.0.
        eps (float): Epsilon for numerical stability. Defaults to 1e-6.
        mode (str): Default "clamp" matches the official GeM operator.
            "abs" explicitly selects the magnitude-pooling variant.

    The exponent is an unconstrained shared parameter, as in the original.
    Callers must keep it away from zero and choose suitable optimizer settings.
    Only the output adapter differs: return ([B,C], auxiliary), rather than
    the original [B,C,1,1] Tensor.
    """

    def __init__(
        self,
        in_channels: int | None = None,
        p: float = 3.0,
        eps: float = 1e-6,
        mode: str = "clamp",
    ) -> None:
        super().__init__(in_channels=in_channels)
        self.out_channels = in_channels
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        if mode not in {"clamp", "abs"}:
            raise ValueError("mode must be 'clamp' or 'abs'")
        self.mode = mode

    def _pow_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "clamp":
            return x.clamp_min(self.eps)
        if self.mode == "abs":
            return x.abs() + self.eps
        raise ValueError(f"mode must be 'clamp' or 'abs', got {self.mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Descriptor [B,C] and auxiliary p [1,1,1], shared across all images.
            The auxiliary value remains differentiable.
        """
        p = self.p
        x_pow = self._pow_input(x).pow(p)
        x_pow = F.avg_pool2d(x_pow, (x_pow.size(-2), x_pow.size(-1)))
        x_pow = x_pow.pow(1.0 / p)

        y = x_pow.flatten(1)
        # Shared parameter for analysis: (1, 1, 1), not sample-specific.
        aux = {"p": p.view(-1, 1, 1)}
        return y, aux
