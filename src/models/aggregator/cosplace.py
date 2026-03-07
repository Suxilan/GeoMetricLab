"""
    Rethinking Visual Geo-Localization for Large-Scale Applications

    Paper: https://arxiv.org/abs/2204.02287
    Code repo: https://github.com/gmberton/CosPlace

    Reference:
    @InProceedings{Berton_CVPR_2022_CosPlace,
        author    = {Berton, Gabriele and Masone, Carlo and Caputo, Barbara},
        title     = {Rethinking Visual Geo-Localization for Large-Scale Applications},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {4878-4888}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any
from .base import AggregatorBase
from .gem import GeM

# Code adapted from CosPlace, MIT license
# https://github.com/gmberton/CosPlace/blob/main/cosplace_model/cosplace_network.py
class CosPlace(AggregatorBase):
    """CosPlace Aggregator.
    
    Combines L2 Normalization, GeM Pooling, and a Linear layer.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. Defaults to 512.
    """
    def __init__(self, in_channels: int, out_channels: int = 512):
        super().__init__(in_channels=in_channels)
        self.gem = GeM(in_channels=in_channels)
        self.fc = nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels
        
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[torch.Tensor, None]: 
                - Global descriptor of shape (B, D).
                - None.
        """
        # x: (B, C, H, W)
        # CosPlace uses L2Norm before GeM
        x = F.normalize(x, p=2, dim=1)
        x, _ = self.gem(x)
        y = self.fc(x)
        return y, None
