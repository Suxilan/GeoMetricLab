from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torchvision.transforms import CenterCrop


class DepthAnythingV2Backbone(nn.Module):
    """Depth-Anything-V2 encoder backbone adapter.

    This wrapper extracts ViT patch/cls features from the encoder part only,
    so it can be used as a drop-in backbone for GeoEncoder aggregators.

    Expected checkpoint formats:
    - full DepthAnythingV2 checkpoint with keys prefixed by `pretrained.`
    - raw encoder checkpoint with ViT keys (e.g. `blocks.*`, `patch_embed.*`)
    """

    AVAILABLE_MODELS = {
        "depth_anything_v2_vits14": "vits",
        "depth_anything_v2_vitb14": "vitb",
        "depth_anything_v2_vitl14": "vitl",
        "depth_anything_v2_vitg14": "vitg",
    }

    EMBED_DIM = {
        "vits": 384,
        "vitb": 768,
        "vitl": 1024,
        "vitg": 1536,
    }

    INTERMEDIATE_LAYER_IDX: dict[str, list[int]] = {
        "vits": [2, 5, 8, 11],
        "vitb": [2, 5, 8, 11],
        "vitl": [4, 11, 17, 23],
        "vitg": [9, 19, 29, 39],
    }

    def __init__(
        self,
        backbone_name: str = "depth_anything_v2_vitb14",
        num_unfrozen_blocks: int = 0,
        checkpoint_path: str | None = None,
        *,
        repository="third_party/Depth-Anything-V2",
        encoder=None,
        pretrained=True,
        native_attention=True,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name.lower().strip()
        if self.backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported backbone: {backbone_name}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        self.encoder_key = self.AVAILABLE_MODELS[self.backbone_name]
        self.patch_size = 14

        if encoder is None:
            repo_root = Path(repository).resolve()
            if not repo_root.is_dir():
                raise FileNotFoundError(
                    f"Provide a Depth-Anything-V2 repository: {repo_root}"
                )
            import sys

            sys.path.insert(0, str(repo_root))
            try:
                from depth_anything_v2.dinov2 import DINOv2

                encoder = DINOv2(self.encoder_key)
                if native_attention:
                    from types import MethodType
                    from depth_anything_v2.dinov2_layers.attention import Attention

                    # Upstream MemEffAttention unconditionally uses xFormers
                    # when installed, including on unsupported CPU inputs.
                    # Bind its native parent forward to the same parameters.
                    for block in encoder.blocks:
                        block.attn.forward = MethodType(Attention.forward, block.attn)
            finally:
                sys.path.remove(str(repo_root))
            self.dino = encoder
            if pretrained:
                self._load_encoder_weights(checkpoint_path)
            elif checkpoint_path is not None:
                raise ValueError("checkpoint_path requires pretrained=True")
        else:
            if checkpoint_path is not None:
                raise ValueError("Choose an injected encoder or checkpoint_path")
            self.dino = encoder
        self.out_channels = getattr(
            self.dino, "embed_dim", self.EMBED_DIM[self.encoder_key]
        )

        self._setup_trainable_blocks(num_unfrozen_blocks)

        logging.getLogger(__name__).info(f"[{self.__class__.__name__}] initialized")
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}]   backbone_name: {self.backbone_name}"
        )
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}]   encoder: {self.encoder_key}"
        )
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}]   out_channels: {self.out_channels}"
        )
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}]   num_unfrozen_blocks: {num_unfrozen_blocks}"
        )

    def _resolve_checkpoint(self, checkpoint_path: str | None) -> Path | None:
        if checkpoint_path:
            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(f"Depth-Anything checkpoint not found: {path}")
            return path

        default_map = {
            "vits": "weights/depth_anything_v2_vits14.pth",
            "vitb": "weights/depth_anything_v2_vitb14.pth",
            "vitl": "weights/depth_anything_v2_vitl14.pth",
            "vitg": "weights/depth_anything_v2_vitg14.pth",
        }
        default_path = Path(default_map[self.encoder_key])
        if default_path.is_file():
            return default_path

        return None

    def _extract_encoder_state_dict(self, checkpoint_obj: dict) -> dict:
        state = checkpoint_obj
        if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
            state = checkpoint_obj["state_dict"]

        if not isinstance(state, dict):
            return {}

        encoder_state = {}
        for k, v in state.items():
            nk = k
            nk = nk.removeprefix("module.")
            nk = nk.removeprefix("model.")

            if nk.startswith("pretrained."):
                encoder_state[nk[len("pretrained.") :]] = v
            elif nk.startswith("backbone."):
                encoder_state[nk[len("backbone.") :]] = v
            else:
                # assume encoder-only checkpoint
                encoder_state[nk] = v

        return encoder_state

    def _load_encoder_weights(self, checkpoint_path: str | None) -> None:
        ckpt = self._resolve_checkpoint(checkpoint_path)
        if ckpt is None:
            raise FileNotFoundError(
                "Provide a Depth-Anything checkpoint or set pretrained=False"
            )

        checkpoint = torch.load(ckpt, map_location="cpu")
        encoder_state = self._extract_encoder_state_dict(checkpoint)
        if not encoder_state:
            raise ValueError(f"No encoder weights found in checkpoint: {ckpt}")
        if not set(encoder_state).intersection(dict(self.dino.named_parameters())):
            raise ValueError(
                f"Checkpoint does not match any encoder parameters: {ckpt}"
            )

        # Partial checkpoints are valid, but must initialize actual parameters.
        missing, unexpected = self.dino.load_state_dict(encoder_state, strict=False)
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}] loaded encoder checkpoint: {ckpt}"
        )
        logging.getLogger(__name__).info(
            f"[{self.__class__.__name__}] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )

    def _setup_trainable_blocks(self, num_unfrozen_blocks: int) -> None:
        # Freeze all by default.
        for p in self.dino.parameters():
            p.requires_grad = False

        if num_unfrozen_blocks <= 0:
            return

        total_blocks = len(self.dino.blocks)
        split = max(total_blocks - int(num_unfrozen_blocks), 0)

        for blk in self.dino.blocks[split:]:
            for p in blk.parameters():
                p.requires_grad = True

        for p in self.dino.norm.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape
        h_new = (h // self.patch_size) * self.patch_size
        w_new = (w // self.patch_size) * self.patch_size
        x = CenterCrop((h_new, w_new))(x)

        layer_ids = [self.INTERMEDIATE_LAYER_IDX[self.encoder_key][-1]]
        outputs = self.dino.get_intermediate_layers(
            x,
            n=layer_ids,
            reshape=True,
            return_class_token=True,
            norm=True,
        )

        # outputs: tuple([(patch_tokens, cls_token)])
        patch_tokens, cls_token = outputs[-1]

        # Keep API aligned with other ViT backbones.
        attn = None
        return cls_token, patch_tokens, attn
