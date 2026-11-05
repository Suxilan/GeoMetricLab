from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import torch
from .local import load_dinov3
from torch import nn
from torchvision.transforms import CenterCrop


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


log = logging.getLogger(__name__)


class PEFTDinoV3(nn.Module):
    """DINOv3 backbone with LoRA (PEFT) support.

    This class keeps output contract aligned with `DinoV3`:
    returns `(cls_token, patch_tokens, attn)` where `patch_tokens` is `(B, C, H, W)`.
    """

    AVAILABLE_MODELS = [
        "peft_dinov3_vits16",
        "peft_dinov3_vits16plus",
        "peft_dinov3_vitb16",
        "peft_dinov3_vitl16",
        "peft_dinov3_vith16plus",
        "peft_dinov3_vit7b16",
    ]

    ARCH_CONFIG = {
        "peft_dinov3_vits16": {
            "depth": 12,
            "embed_dim": 384,
            "patch_size": 16,
            "base": "dinov3_vits16",
        },
        "peft_dinov3_vits16plus": {
            "depth": 12,
            "embed_dim": 384,
            "patch_size": 16,
            "base": "dinov3_vits16plus",
        },
        "peft_dinov3_vitb16": {
            "depth": 12,
            "embed_dim": 768,
            "patch_size": 16,
            "base": "dinov3_vitb16",
        },
        "peft_dinov3_vitl16": {
            "depth": 24,
            "embed_dim": 1024,
            "patch_size": 16,
            "base": "dinov3_vitl16",
        },
        "peft_dinov3_vith16plus": {
            "depth": 32,
            "embed_dim": 1280,
            "patch_size": 16,
            "base": "dinov3_vith16plus",
        },
        "peft_dinov3_vit7b16": {
            "depth": 40,
            "embed_dim": 4096,
            "patch_size": 16,
            "base": "dinov3_vit7b16",
        },
    }

    MODEL_WEIGHTS_MAP = {
        "peft_dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "peft_dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "peft_dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "peft_dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "peft_dinov3_vith16plus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "peft_dinov3_vit7b16": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    }

    def __init__(
        self,
        backbone_name: str = "peft_dinov3_vitb16",
        num_unfrozen_blocks: int = 1,
        num_layers: int | None = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_targets: Sequence[str] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
        *,
        repository="third_party/dinov3",
        weights=None,
        pretrained=True,
        dino=None,
        capture_attention=False,
    ):
        super().__init__()

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Backbone {backbone_name} is not recognized! Supported backbones: {self.AVAILABLE_MODELS}"
            )

        cfg = self.ARCH_CONFIG[backbone_name]
        self.backbone_name = backbone_name
        self.base_backbone_name = cfg["base"]
        self.total_layers = cfg["depth"]
        self.patch_size = cfg["patch_size"]

        self.num_unfrozen_blocks = int(num_unfrozen_blocks)
        self.num_layers = (
            int(num_layers) if num_layers is not None else self.total_layers
        )
        if self.num_layers > self.total_layers:
            raise ValueError(
                f"num_layers ({self.num_layers}) cannot be greater than total layers ({self.total_layers})"
            )

        log.debug("PEFT DINOv3 initialized with:")
        log.debug(f"  backbone_name: {self.backbone_name}")
        log.debug(f"  base_backbone_name: {self.base_backbone_name}")
        log.debug(f"  num_unfrozen_blocks: {self.num_unfrozen_blocks}")
        log.debug(f"  num_layers: {self.num_layers}/{self.total_layers}")

        repo_dir = str(repository)
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        weight_name = self.MODEL_WEIGHTS_MAP[self.backbone_name]
        weights_path = (
            str(weights)
            if weights is not None
            else os.path.join(cache_dir, weight_name)
        )
        if dino is None and pretrained and not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Weights not found at {weights_path}. Please download them first."
            )

        self.dino = (
            dino
            if dino is not None
            else load_dinov3(
                repo_dir,
                self.base_backbone_name,
                pretrained=pretrained,
                **({"weights": weights_path} if pretrained else {}),
            )
        )

        # Freeze everything first.
        for p in self.dino.parameters():
            p.requires_grad = False

        self.split = max(0, self.num_layers - self.num_unfrozen_blocks)

        self._inject_lora(
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            lora_bias=str(lora_bias),
            lora_targets=list(lora_targets),
        )

        # Keep final norms trainable for stable PEFT adaptation.
        if self.num_layers == self.total_layers:
            if hasattr(self.dino, "norm") and hasattr(self.dino.norm, "parameters"):
                for p in self.dino.norm.parameters():
                    p.requires_grad = True
            if hasattr(self.dino, "cls_norm") and hasattr(
                self.dino.cls_norm, "parameters"
            ):
                for p in self.dino.cls_norm.parameters():
                    p.requires_grad = True

        self.out_channels = self.dino.embed_dim
        self.n_storage_tokens = getattr(self.dino, "n_storage_tokens", 0)

        self._qkv_out = None
        if capture_attention and self.num_layers > 0:
            target_layer = self.dino.blocks[self.num_layers - 1]
            target_layer.attn.qkv.register_forward_hook(self._hook_qkv)

        self._print_trainable_summary()

    def _collect_target_module_names(self, lora_targets: list[str]) -> list[str]:
        block_indices = list(range(self.split, self.num_layers))
        prefixes = [f"blocks.{i}." for i in block_indices]

        full_names: list[str] = []
        for name, module in self.dino.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.startswith(p) for p in prefixes):
                continue
            if any(name.endswith(t) for t in lora_targets):
                full_names.append(name)

        return sorted(set(full_names))

    def _inject_lora(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_bias: str,
        lora_targets: list[str],
    ) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "PEFT is required for PEFTDinoV3. Please install: `pip install peft`"
            ) from exc

        target_modules = self._collect_target_module_names(lora_targets)
        if len(target_modules) == 0:
            raise ValueError(
                "No target modules found for LoRA injection. "
                f"split={self.split}, num_layers={self.num_layers}, targets={lora_targets}"
            )

        log.debug(f"  LoRA targets ({len(target_modules)}):")
        for n in target_modules[:24]:
            log.debug(f"    - {n}")
        if len(target_modules) > 24:
            log.debug(f"    ... (+{len(target_modules) - 24} more)")

        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=target_modules,
        )

        _ = get_peft_model(self.dino, peft_cfg)

        for name, p in self.dino.named_parameters():
            p.requires_grad = "lora_" in name

    def _print_trainable_summary(self) -> None:
        total = sum(p.numel() for p in self.dino.parameters())
        trainable = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        ratio = 100.0 * trainable / max(total, 1)
        log.debug(f"  trainable params: {trainable}/{total} ({ratio:.4f}%)")

    def _hook_qkv(self, module, input, output) -> None:
        self._qkv_out = output

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape

        h_new = (h // self.patch_size) * self.patch_size
        w_new = (w // self.patch_size) * self.patch_size
        x = CenterCrop((h_new, w_new))(x)

        x, (h_feat, w_feat) = self.dino.prepare_tokens_with_masks(x)

        rope_sincos = None
        if hasattr(self.dino, "rope_embed") and self.dino.rope_embed is not None:
            rope_sincos = self.dino.rope_embed(H=h_feat, W=w_feat)

        with torch.no_grad():
            for blk in self.dino.blocks[: self.split]:
                x = blk(x, rope_sincos)

        x = x.detach()

        for blk in self.dino.blocks[self.split : self.num_layers]:
            x = blk(x, rope_sincos)

        if self.num_layers == self.total_layers:
            if getattr(self.dino, "untie_cls_and_patch_norms", False):
                n_special = 1 + self.n_storage_tokens
                x_special = x[:, :n_special]
                x_patches = x[:, n_special:]
                if hasattr(self.dino, "cls_norm"):
                    x_special = self.dino.cls_norm(x_special)
                if hasattr(self.dino, "norm"):
                    x_patches = self.dino.norm(x_patches)
                x = torch.cat([x_special, x_patches], dim=1)
            else:
                if hasattr(self.dino, "norm"):
                    x = self.dino.norm(x)

        cls_token = x[:, 0]

        n_patches = h_feat * w_feat
        patch_tokens = x[:, -n_patches:, :]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(
            b, self.out_channels, h_feat, w_feat
        )

        attn = None
        if self._qkv_out is not None:
            qkv = self._qkv_out
            self._qkv_out = None

            b_tok, n_tok, _ = qkv.shape
            last_blk = self.dino.blocks[self.num_layers - 1]
            num_heads = last_blk.attn.num_heads
            head_dim = self.out_channels // num_heads
            scale = head_dim**-0.5

            qkv = qkv.reshape(b_tok, n_tok, 3, num_heads, head_dim)
            q, k, _ = qkv.unbind(2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

            if rope_sincos is not None:
                sin, cos = rope_sincos
                sin = sin.to(q.device)
                cos = cos.to(q.device)
                prefix = n_tok - sin.shape[0]
                if prefix >= 0:
                    q_patches = q[:, :, prefix:, :]
                    k_patches = k[:, :, prefix:, :]
                    q_patches = rope_apply(q_patches, sin, cos)
                    k_patches = rope_apply(k_patches, sin, cos)
                    q = torch.cat((q[:, :, :prefix, :], q_patches), dim=2)
                    k = torch.cat((k[:, :, :prefix, :], k_patches), dim=2)

            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)

        return cls_token, patch_tokens, attn
