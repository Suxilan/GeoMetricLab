"""timm ResNet feature maps with explicit frozen-prefix execution.

Adapted from Amar Ali-bey's Bag-of-Queries:
https://github.com/amaralibey/Bag-of-Queries
"""

# Copyright (c) 2024 Amar Ali-bey
# MIT license; see LICENSES/BoQ.txt and NOTICE.

import logging

import torch
from torch import nn


class ResNet(nn.Module):
    """Map images [N,3,H,W] to the selected final ResNet feature map.

    With pretrained=True, only the last unfreeze_n_blocks modules are trainable.
    The frozen prefix runs under no_grad, preserving the original BN lifecycle.
    With pretrained=False all blocks train, independent of unfreeze_n_blocks.
    crop_last_block omits layer4. Channel discovery never runs a dummy forward.
    """

    MODEL_WEIGHTS_MAP = {
        "resnet18": "resnet18.a1_in1k",
        "resnet34": "resnet34.a1_in1k",
        "resnet50": "resnet50.a1_in1k",
        "resnet101": "resnet101.a1_in1k",
        "resnet152": "resnet152.a1_in1k",
        "resnext50": "resnext50_32x4d.a1_in1k",
    }
    AVAILABLE_MODELS = tuple(MODEL_WEIGHTS_MAP)

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        unfreeze_n_blocks: int = 1,
        crop_last_block: bool = True,
    ) -> None:
        super().__init__()
        import timm

        if backbone_name not in self.MODEL_WEIGHTS_MAP:
            raise ValueError(
                f"Unsupported ResNet {backbone_name!r}: {self.AVAILABLE_MODELS}"
            )
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block
        model = timm.create_model(
            self.MODEL_WEIGHTS_MAP[backbone_name],
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.net = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu if hasattr(model, "relu") else model.act1,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            *([] if crop_last_block else [model.layer4]),
        )
        if not isinstance(unfreeze_n_blocks, int) or not 0 <= unfreeze_n_blocks <= len(
            self.net
        ):
            raise ValueError(
                f"unfreeze_n_blocks must be an integer in [0,{len(self.net)}]"
            )
        self.frozen_layer_count = len(self.net) - unfreeze_n_blocks if pretrained else 0
        for layer in self.net[: self.frozen_layer_count]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
        if not pretrained and unfreeze_n_blocks:
            logging.getLogger(__name__).info(
                "Randomly initialized ResNet trains all blocks"
            )
            self.unfreeze_n_blocks = 0
        feature_index = 3 if crop_last_block else 4
        self.out_channels = model.feature_info[feature_index]["num_chs"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = images
        if self.pretrained and self.frozen_layer_count:
            with torch.no_grad():
                for layer in self.net[: self.frozen_layer_count]:
                    features = layer(features)
            features = features.detach()
        for layer in self.net[self.frozen_layer_count :]:
            features = layer(features)
        return features
