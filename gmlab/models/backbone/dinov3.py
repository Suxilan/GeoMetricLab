"""Local DINOv3 feature maps with explicit frozen-prefix and attention semantics."""

from pathlib import Path

import torch
from .local import load_dinov3
from torch import nn
from torchvision.transforms.functional import center_crop


class DinoV3(nn.Module):
    """Return (CLS [B,C], patches [B,C,H/P,W/P], attention or None).

    Images are center-cropped to patch-size multiples. The frozen block prefix
    runs without gradients, then tokens are detached; only the active tail and
    final patch norm train, preserving the former GeoMetricLab backbone math.
    With pretrained=True only an existing local weight file is used. Supplying
    dino injects a constructed DINO module and performs no repository/weight IO.
    capture_attention=True reconstructs the former RoPE attention output; false
    avoids retaining QKV and the quadratic attention matrix during extraction.
    """

    MODEL_WEIGHTS_MAP = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    }

    def __init__(
        self,
        backbone_name="dinov3_vitl16",
        num_unfrozen_blocks=0,
        num_layers=None,
        *,
        repository="third_party/dinov3",
        pretrained=True,
        weights=None,
        capture_attention=True,
        dino=None,
    ):
        super().__init__()
        if backbone_name not in self.MODEL_WEIGHTS_MAP:
            raise ValueError(f"Unsupported DINOv3 backbone: {backbone_name}")
        if dino is not None and weights is not None:
            raise ValueError("Choose an injected DINO module or local weights")
        if not pretrained and weights is not None:
            raise ValueError("weights requires pretrained=True")
        if dino is None:
            repository = Path(repository).resolve()
            if not (repository / "hubconf.py").is_file():
                raise FileNotFoundError(f"DINOv3 hubconf.py not found in {repository}")
            options = {"pretrained": pretrained}
            if pretrained:
                weights = (
                    Path(weights)
                    if weights is not None
                    else (
                        Path.home()
                        / ".cache/torch/hub/checkpoints"
                        / self.MODEL_WEIGHTS_MAP[backbone_name]
                    )
                )
                if not weights.is_file():
                    raise FileNotFoundError(
                        f"DINOv3 local weights not found: {weights}"
                    )
                options["weights"] = str(weights.resolve())
            dino = load_dinov3(repository, backbone_name, **options)
        self.dino = dino
        self.backbone_name = backbone_name
        self.total_layers = len(dino.blocks)
        self.num_layers = self.total_layers if num_layers is None else num_layers
        if (
            type(self.num_layers) is not int
            or not 1 <= self.num_layers <= self.total_layers
        ):
            raise ValueError(
                "num_layers must select a nonempty prefix of DINOv3 blocks"
            )
        if (
            type(num_unfrozen_blocks) is not int
            or not 0 <= num_unfrozen_blocks <= self.num_layers
        ):
            raise ValueError("num_unfrozen_blocks must be in [0,num_layers]")
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.split = self.num_layers - num_unfrozen_blocks
        self.patch_size = int(dino.patch_size)
        self.out_channels = int(dino.embed_dim)
        self.n_storage_tokens = int(dino.n_storage_tokens)
        self.capture_attention = capture_attention
        self._qkv_out = None
        for parameter in dino.parameters():
            parameter.requires_grad_(False)
        for block in dino.blocks[self.split : self.num_layers]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        if self.num_layers == self.total_layers:
            for parameter in dino.norm.parameters():
                parameter.requires_grad_(True)
        if capture_attention:
            dino.blocks[self.num_layers - 1].attn.qkv.register_forward_hook(
                self._capture_qkv
            )

    def _capture_qkv(self, module, inputs, output):
        self._qkv_out = output

    def _normalize_tokens(self, tokens):
        if self.num_layers != self.total_layers:
            return tokens
        if not self.dino.untie_cls_and_patch_norms:
            return self.dino.norm(tokens)
        special = 1 + self.n_storage_tokens
        return torch.cat(
            [
                self.dino.cls_norm(tokens[:, :special]),
                self.dino.norm(tokens[:, special:]),
            ],
            dim=1,
        )

    def _attention(self, rope):
        qkv, self._qkv_out = self._qkv_out, None
        if qkv is None:
            return None
        batch, count, _ = qkv.shape
        heads = self.dino.blocks[self.num_layers - 1].attn.num_heads
        width = self.out_channels // heads
        q, k, _ = qkv.reshape(batch, count, 3, heads, width).unbind(2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        if rope is not None:
            sin, cos = (value.to(q.device) for value in rope)
            if sin.ndim != 2 or cos.shape != sin.shape:
                raise ValueError(
                    "Attention reconstruction requires shared [patches,head_dim] RoPE"
                )
            prefix = count - sin.shape[0]
            if prefix < 0:
                raise ValueError("RoPE patch count exceeds token count")

            def rotate(values):
                patches = values[:, :, prefix:]
                left, right = patches.chunk(2, dim=-1)
                rotated = torch.cat([-right, left], dim=-1)
                return torch.cat(
                    [values[:, :, :prefix], patches * cos + rotated * sin], dim=2
                )

            q, k = rotate(q), rotate(k)
        return ((q @ k.transpose(-2, -1)) * width**-0.5).softmax(dim=-1)

    def forward(self, images):
        self._qkv_out = None
        batch, _, height, width = images.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        if not grid_h or not grid_w:
            raise ValueError("DINOv3 input needs at least one full patch per axis")
        images = center_crop(
            images, [grid_h * self.patch_size, grid_w * self.patch_size]
        )
        tokens, (grid_h, grid_w) = self.dino.prepare_tokens_with_masks(images)
        rope = (
            self.dino.rope_embed(H=grid_h, W=grid_w)
            if getattr(self.dino, "rope_embed", None) is not None
            else None
        )
        with torch.no_grad():
            for block in self.dino.blocks[: self.split]:
                tokens = block(tokens, rope)
        tokens = tokens.detach()
        for block in self.dino.blocks[self.split : self.num_layers]:
            tokens = block(tokens, rope)
        tokens = self._normalize_tokens(tokens)
        patches = tokens[:, -(grid_h * grid_w) :].transpose(1, 2)
        patches = patches.reshape(batch, self.out_channels, grid_h, grid_w)
        return tokens[:, 0], patches, self._attention(rope)
