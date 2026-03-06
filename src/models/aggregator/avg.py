import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any
from .base import AggregatorBase

class Avg(AggregatorBase):
    """Global Average Pooling (GAP) Aggregator.
    
    Aggregates local features by averaging them across spatial dimensions.
    """
    def __init__(
        self, 
        in_channels: 
        int = None):
        super().__init__(in_channels=in_channels)
        self.out_channels = in_channels
        
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) or (B, N, C).

        Returns:
            Tuple[torch.Tensor, None]: 
                - Global descriptor of shape (B, C).
                - None (no auxiliary info).
        """
        if x.dim() == 4:
            # x: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            y = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        elif x.dim() == 3:
            # x: (B, N, C) -> (B, C)
            y = x.mean(dim=1)
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {x.dim()}D")
            
        return y, None
    