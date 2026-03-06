import torch
import torch.nn as nn
import timm
from src.utils.logger import print_rank_0


class SwinTransformer(nn.Module):
    AVAILABLE_MODELS = [
        "swin_v2t",
        "swin_v2s",
        "swin_v2b",
        "swin_v2l",
    ]

    MODEL_WEIGHTS_MAP = {
        "swin_v2t": "swinv2_tiny_window16_256.ms_in1k",
        "swin_v2s": "swinv2_small_window16_256.ms_in1k",
        "swin_v2b": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        "swin_v2l": "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    }

    ARCH_CONFIG = {
        "swin_v2t": {
            "depth": 12,
            "embed_dim": 96,
            "patch_size": 4,
            "stage_channel_dims": [96, 192, 384, 768],
            "blocks_per_stage": [2, 2, 6, 2],
        },
        "swin_v2s": {
            "depth": 24,
            "embed_dim": 96,
            "patch_size": 4,
            "stage_channel_dims": [96, 192, 384, 768],
            "blocks_per_stage": [2, 2, 18, 2],
        },
        "swin_v2b": {
            "depth": 24,
            "embed_dim": 128,
            "patch_size": 4,
            "stage_channel_dims": [128, 256, 512, 1024],
            "blocks_per_stage": [2, 2, 18, 2],
        },
        "swin_v2l": {
            "depth": 24,
            "embed_dim": 192,
            "patch_size": 4,
            "stage_channel_dims": [192, 384, 768, 1536],
            "blocks_per_stage": [2, 2, 18, 2],
        },
    }

    def print(self, msg):
        print_rank_0(f"[{self.__class__.__name__}] {msg}")

    def __init__(self, backbone_name="swin_v2b", pretrained=True, num_unfrozen_blocks=0):
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.num_unfrozen_blocks = min(
            num_unfrozen_blocks,
            len(self.ARCH_CONFIG.get(backbone_name, {}).get("stage_channel_dims", [])),
        )

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized! Supported: {self.AVAILABLE_MODELS}")

        self.arch = self.ARCH_CONFIG[backbone_name]
        self.num_unfrozen_blocks = min(num_unfrozen_blocks, len(self.arch["stage_channel_dims"]))

        model_weight = self.MODEL_WEIGHTS_MAP[backbone_name]
        arch_tag = model_weight.split(".")[0]
        tag = model_weight.split(".")[-1] if pretrained else ""

        self.model = timm.create_model(
            model_weight,
            pretrained=pretrained,
            num_classes=0,
            strict_img_size=False,
        )

        self.feature_info = self.model.feature_info
        self.out_channels = self.feature_info[-1]["num_chs"]
        self.num_stages = len(self.model.layers)

        self.num_unfrozen_blocks = min(self.num_unfrozen_blocks, self.num_stages)
        self.split_idx = self.num_stages - self.num_unfrozen_blocks

        self.print(f"Initializing {backbone_name} (timm: {arch_tag})")
        self.print(f"  pretrained: {pretrained}, tag: {tag}")
        self.print(f"  Total Stages: {self.num_stages}")
        self.print(f"  Freezing first {self.split_idx} stages (unfreezing last {self.num_unfrozen_blocks})")

        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

            for layer in self.model.layers[self.split_idx:]:
                for param in layer.parameters():
                    param.requires_grad = True

            for param in self.model.norm.parameters():
                param.requires_grad = True

            if self.split_idx == 0 and hasattr(self.model, "patch_embed"):
                for param in self.model.patch_embed.parameters():
                    param.requires_grad = True

    def forward(self, x):
        # === Part 1: Frozen prefix (patch_embed + initial stages) ===
        # Run patch_embed under no_grad if it's in the frozen prefix
        if hasattr(self.model, "patch_embed"):
            if self.pretrained and self.split_idx > 0:
                with torch.no_grad():
                    x = self.model.patch_embed(x)
            else:
                x = self.model.patch_embed(x)

        # Frozen stages
        if self.pretrained and self.split_idx > 0:
            with torch.no_grad():
                for i in range(0, self.split_idx):
                    x = self.model.layers[i](x)
            # explicitly break gradient flow from the frozen prefix
            x = x.detach()

        # === Part 2: Unfrozen remainder ===
        for i in range(self.split_idx, self.num_stages):
            x = self.model.layers[i](x)

        # Final norm
        x = self.model.norm(x)

        # Output formatting: (B, H, W, C) -> (B, C, H, W)
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)

        return x
