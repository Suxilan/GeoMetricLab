import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
from .base import AggregatorBase

class CLS(AggregatorBase):
    """CLS Token Aggregator.
    
    Returns the CLS token from the backbone output.
    """
    def __init__(self, in_channels: int = None):
        super().__init__(in_channels=in_channels)
        self.out_channels = in_channels
        
    def forward(self, x: torch.Tensor, cls_token: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, Optional[Any]]:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor (unused).
            cls_token (Optional[torch.Tensor]): CLS token of shape (B, C).

        Returns:
            Tuple[torch.Tensor, None]: 
                - Global descriptor (CLS token).
                - None.
        """
        if cls_token is None:
            raise ValueError("CLSAggregator requires a CLS token input (e.g. from ViT backbone).")
        return cls_token, None
