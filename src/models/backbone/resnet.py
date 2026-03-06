# adapted from Amar Ali-bey's Bag-of-Queries
# ----------------------------------------------------------------------------
# https://github.com/amaralibey/Bag-of-Queries


import torch
import torch.nn as nn
import timm
from src.utils.logger import print_rank_0

class ResNet(nn.Module):
    AVAILABLE_MODELS = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50",
    ]

    MODEL_WEIGHTS_MAP = {
        "resnet18": "resnet18.a1_in1k",
        "resnet34": "resnet34.a1_in1k",
        "resnet50": "resnet50.a1_in1k",
        "resnet101": "resnet101.a1_in1k",
        "resnet152": "resnet152.a1_in1k",
        "resnext50": "resnext50_32x4d.a1_in1k",
    }

    def print(self, msg):
        print_rank_0(f"[{self.__class__.__name__}] {msg}")

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        unfreeze_n_blocks=1,
        crop_last_block=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized! " 
                             f"Supported backbones are: {self.AVAILABLE_MODELS}")

        # Load the model
        model_weight = self.MODEL_WEIGHTS_MAP[backbone_name]
        self.print(f"Initializing {backbone_name} (timm: {model_weight})")
        
        resnet = timm.create_model(
            model_weight,
            pretrained=pretrained,
            num_classes=0, 
            global_pool=''
        )
        
        # Compatibility with torchvision structure for self.net construction
        if not hasattr(resnet, 'relu'):
            resnet.relu = resnet.act1

        # Create backbone with only the necessary layers
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            *([] if crop_last_block else [resnet.layer4]),
        )

        # Handle trainable/frozen layers
        nb_layers = len(self.net)
        assert (
            isinstance(unfreeze_n_blocks, int) and 0 <= unfreeze_n_blocks <= nb_layers
        ), f"unfreeze_n_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if pretrained:
            # Freeze required layers
            self.frozen_layer_count = max(0, nb_layers - unfreeze_n_blocks)
            for layer in self.net[: self.frozen_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks is ignored when pretrained=False. Setting it to 0.")
                self.unfreeze_n_blocks = 0
            self.frozen_layer_count = 0

        # Output channels
        if self.crop_last_block:
            last_layer = resnet.layer3
        else:
            last_layer = resnet.layer4

        # timm layers are Sequentials. We need to inspect the last block.
        # For ResNet, the last block is the last element of the layer Sequential.
        # It usually has conv3 (bottleneck) or conv2 (basic).
        # We can use a dummy forward pass to be robust.
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            # We can run just the net
            out = self.net(dummy_input)
            self.out_channels = out.shape[1]

    def forward(self, x):
        # Frozen prefix
        out = x
        if self.pretrained and self.frozen_layer_count > 0:
            with torch.no_grad():
                for idx in range(0, self.frozen_layer_count):
                    out = self.net[idx](out)
            out = out.detach()

        # Remaining (trainable) layers
        for idx in range(self.frozen_layer_count, len(self.net)):
            out = self.net[idx](out)

        return out