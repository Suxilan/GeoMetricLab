import torch
import torch.nn as nn
from typing import Tuple, Any, Optional

class AggregatorBase(nn.Module):
    """Base class for all aggregators in GeoMetricLab.
    
    All aggregators must implement the forward method which returns a tuple:
    (global_descriptor, auxiliary_info).
    """
    def __init__(self, in_channels: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = None  # To be defined in subclasses
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input features.
            **kwargs: Additional arguments (e.g., cls_token).

        Returns:
            Tuple[torch.Tensor, Optional[Any]]: 
                - Global descriptor (B, D).
                - Auxiliary info (e.g., attention maps) or None.
        """
        raise NotImplementedError
