"""
    EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition

    Paper: https://arxiv.org/abs/2308.10832
    Code repo: https://github.com/gmberton/EigenPlaces

    Reference:
    @inproceedings{Berton_2023_EigenPlaces,
        title={EigenPlaces: Training Viewpoint Robust Models for Visual Place Recognition},
        author={Berton, Gabriele and Trivigno, Gabriele and Caputo, Barbara and Masone, Carlo},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        year={2023},
        month={October},
        pages={11080-11090}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any

from .base import AggregatorBase
from .gem import GeM


# Code adapted from EigenPlaces, MIT license
# https://github.com/gmberton/EigenPlaces/blob/main/eigenplaces_model/eigenplaces_network.py
class EigenPlace(AggregatorBase):
    """EigenPlace-style head: L2Norm -> GeM -> Linear.

    Reference snippet (EigenPlace):
        L2Norm(), GeM(), Flatten(), Linear(features_dim, fc_output_dim), L2Norm()

    In this codebase, the GeoEncoder applies BN and (optionally) a final L2 norm
    after the aggregator, so we only keep the pre-GeM L2 normalization here.

    Expected input:
      - CNN feature map: (B, C, H, W)

    Returns:
      - descriptor: (B, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int = 512, p: float = 3.0, eps: float = 1e-6):
        super().__init__(in_channels=in_channels)
        self.gem = GeM(in_channels=in_channels, p=p, eps=eps)
        self.fc = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
        if x.dim() != 4:
            raise ValueError(
                f"EigenPlace expects CNN feature maps (B,C,H,W). Got shape {tuple(x.shape)}"
            )

        x = F.normalize(x, p=2, dim=1)
        x, _ = self.gem(x)
        y = self.fc(x)
        return y, None
