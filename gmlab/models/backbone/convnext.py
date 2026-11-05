import logging
import os

import torch
from .local import load_dinov3
from torch import nn


log = logging.getLogger(__name__)


class ConvNeXt(nn.Module):
    """Return DINOv3 ConvNeXt maps [N,C,H/32,W/32] from RGB inputs.

    repository and weights select external assets explicitly. An injected model
    avoids asset IO. num_unfrozen_blocks counts tail stages for pretrained models;
    random initialization trains every stage. No input normalization is applied.
    """

    AVAILABLE_MODELS = [
        "dinov3_convnext_tiny",
        "dinov3_convnext_small",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
    ]

    MODEL_WEIGHTS_MAP = {
        "dinov3_convnext_tiny": "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
        "dinov3_convnext_small": "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
        "dinov3_convnext_base": "dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
        "dinov3_convnext_large": "dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
    }

    # Depths/Dims for reference, though we load from hub
    ARCH_CONFIG = {
        "dinov3_convnext_tiny": dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
        "dinov3_convnext_small": dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
        "dinov3_convnext_base": dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
        "dinov3_convnext_large": dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]),
    }

    def __init__(
        self,
        backbone_name="dinov3_convnext_base",
        pretrained=True,
        num_unfrozen_blocks=1,
        *,
        repository="third_party/dinov3",
        weights=None,
        model=None,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.num_unfrozen_blocks = num_unfrozen_blocks

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Backbone {backbone_name} is not recognized! Supported: {self.AVAILABLE_MODELS}"
            )

        # Determine paths
        repo_dir = str(repository)
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        weight_filename = self.MODEL_WEIGHTS_MAP[backbone_name]
        weights_path = (
            str(weights)
            if weights is not None
            else os.path.join(cache_dir, weight_filename)
        )

        if model is None and pretrained and not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. Please download them first."
            )

        log.debug(f"Initializing {backbone_name}")
        log.debug(f"  pretrained: {pretrained}")
        log.debug(f"  weights: {weights_path if pretrained else 'None'}")

        # Load model from local hub
        # We pass weights path if pretrained is True.
        # The hubconf function signature is: dinov3_convnext_*(pretrained=True, weights=..., **kwargs)
        if model is not None:
            self.model = model
        elif pretrained:
            self.model = load_dinov3(
                repo_dir,
                backbone_name,
                pretrained=True,
                weights=weights_path,
            )
        else:
            self.model = load_dinov3(repo_dir, backbone_name, pretrained=False)

        # Expose layers for split forward
        self.downsample_layers = self.model.downsample_layers
        self.stages = self.model.stages
        self.norm = self.model.norm

        self.num_stages = len(self.stages)  # Should be 4
        self.out_channels = self.model.embed_dim

        # ==== Freezing Logic ====
        # We treat num_unfrozen_blocks as number of stages to unfreeze (0 to 4)
        if pretrained:
            self.split_idx = max(0, self.num_stages - num_unfrozen_blocks)
        else:
            self.split_idx = 0

        log.debug(
            f"  num_unfrozen_blocks: {num_unfrozen_blocks} (Total Stages: {self.num_stages})"
        )
        log.debug(f"  Freezing first {self.split_idx} stages")

        if pretrained:
            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze requested stages
            for i in range(self.split_idx, self.num_stages):
                for param in self.downsample_layers[i].parameters():
                    param.requires_grad = True
                for param in self.stages[i].parameters():
                    param.requires_grad = True

            # Unfreeze final norm if last stage is unfrozen
            if self.split_idx < self.num_stages:
                for param in self.norm.parameters():
                    param.requires_grad = True

    def forward(self, x):
        # Frozen prefix
        if self.pretrained and self.split_idx > 0:
            with torch.no_grad():
                for i in range(self.split_idx):
                    x = self.downsample_layers[i](x)
                    x = self.stages[i](x)
            x = x.detach()

        # Unfrozen stages
        for i in range(self.split_idx, self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # Final output (B, C, H, W)
        # Apply final norm (channels_last)
        x = x.permute(0, 2, 3, 1)  # (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (N, C, H, W)

        return x
