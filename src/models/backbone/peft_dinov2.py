from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

from src.utils.logger import print_rank_0


class PEFTDinoV2(nn.Module):
    """DINOv2 backbone with LoRA (PEFT) support.

    This class keeps GeoMetricLab's output contract identical to `DinoV2`:
    returns `(cls_token, patch_tokens, attn)` where `patch_tokens` is
    `(B, C, H, W)`.

    Notes:
    - For now this class focuses on LoRA with PEFT.
    - To keep behavior aligned with existing training strategy, we still use
      `num_unfrozen_blocks` as the active trainable depth range and execute the
      frozen prefix under `torch.no_grad()`.
    """

    AVAILABLE_MODELS = [
        "peft_dinov2_vits14",
        "peft_dinov2_vitb14",
        "peft_dinov2_vitl14",
        "peft_dinov2_vitg14",
        "peft_dinov2_vits14_reg",
        "peft_dinov2_vitb14_reg",
        "peft_dinov2_vitl14_reg",
        "peft_dinov2_vitg14_reg",
    ]

    ARCH_CONFIG = {
        "peft_dinov2_vits14": {"depth": 12, "embed_dim": 384, "patch_size": 14, "base": "dinov2_vits14"},
        "peft_dinov2_vitb14": {"depth": 12, "embed_dim": 768, "patch_size": 14, "base": "dinov2_vitb14"},
        "peft_dinov2_vitl14": {"depth": 24, "embed_dim": 1024, "patch_size": 14, "base": "dinov2_vitl14"},
        "peft_dinov2_vitg14": {"depth": 40, "embed_dim": 1536, "patch_size": 14, "base": "dinov2_vitg14"},
        "peft_dinov2_vits14_reg": {"depth": 12, "embed_dim": 384, "patch_size": 14, "base": "dinov2_vits14_reg"},
        "peft_dinov2_vitb14_reg": {"depth": 12, "embed_dim": 768, "patch_size": 14, "base": "dinov2_vitb14_reg"},
        "peft_dinov2_vitl14_reg": {"depth": 24, "embed_dim": 1024, "patch_size": 14, "base": "dinov2_vitl14_reg"},
        "peft_dinov2_vitg14_reg": {"depth": 40, "embed_dim": 1536, "patch_size": 14, "base": "dinov2_vitg14_reg"},
    }

    def print(self, msg: str):
        print_rank_0(f"[{self.__class__.__name__}] {msg}")

    def __init__(
        self,
        backbone_name: str = "peft_dinov2_vitb14",
        num_unfrozen_blocks: int = 1,
        num_layers: int | None = None,
        # PEFT / LoRA args
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_targets: Sequence[str] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
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
        self.num_layers = int(num_layers) if num_layers is not None else self.total_layers
        if self.num_layers > self.total_layers:
            raise ValueError(f"num_layers ({self.num_layers}) cannot be greater than total layers ({self.total_layers})")

        self.print("PEFT DINOv2 initialized with:")
        self.print(f"  backbone_name: {self.backbone_name}")
        self.print(f"  base_backbone_name: {self.base_backbone_name}")
        self.print(f"  num_unfrozen_blocks: {self.num_unfrozen_blocks}")
        self.print(f"  num_layers: {self.num_layers}/{self.total_layers}")

        # 1) load base DINOv2
        self.dino = torch.hub.load("third_party/dinov2", self.base_backbone_name, source="local")

        # 2) freeze all params first
        for p in self.dino.parameters():
            p.requires_grad = False

        # 3) determine active trainable tail blocks
        self.split = max(0, self.num_layers - self.num_unfrozen_blocks)

        # 4) inject LoRA on selected blocks/modules
        self._inject_lora(
            lora_r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            lora_bias=str(lora_bias),
            lora_targets=list(lora_targets),
        )

        # Make final norm always trainable for PEFT setups (helpful for scale)
        for param in self.dino.norm.parameters():
            param.requires_grad = True

        self.out_channels = self.dino.embed_dim
        self.num_register_tokens = self.dino.num_register_tokens

        # Hook for attention extraction
        self._qkv_out = None
        if self.num_layers > 0:
            target_layer = self.dino.blocks[self.num_layers - 1]
            target_layer.attn.qkv.register_forward_hook(self._hook_qkv)

        self._print_trainable_summary()

    def _collect_target_module_names(self, lora_targets: List[str]) -> List[str]:
        """Collect full module names to LoRA-ize only in trainable tail blocks."""
        block_indices = list(range(self.split, self.num_layers))
        prefixes = [f"blocks.{i}." for i in block_indices]

        full_names: List[str] = []
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
        lora_targets: List[str],
    ) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:
            raise ImportError(
                "PEFT is required for PEFTDinoV2. Please install: `pip install peft`"
            ) from exc

        target_modules = self._collect_target_module_names(lora_targets)
        if len(target_modules) == 0:
            raise ValueError(
                "No target modules found for LoRA injection. "
                f"split={self.split}, num_layers={self.num_layers}, targets={lora_targets}"
            )

        self.print(f"  LoRA targets ({len(target_modules)}):")
        for n in target_modules[:24]:
            self.print(f"    - {n}")
        if len(target_modules) > 24:
            self.print(f"    ... (+{len(target_modules)-24} more)")

        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=target_modules,
        )

        # `get_peft_model` returns a wrapper, but LoRA modules are injected into
        # the underlying base model object. We keep using `self.dino` APIs below.
        _ = get_peft_model(self.dino, peft_cfg)

        # ensure only LoRA params are trainable (plus optional norm handled outside)
        for name, p in self.dino.named_parameters():
            p.requires_grad = ("lora_" in name)

    def _print_trainable_summary(self) -> None:
        total = sum(p.numel() for p in self.dino.parameters())
        trainable = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        ratio = (100.0 * trainable / max(total, 1))
        self.print(f"  trainable params: {trainable}/{total} ({ratio:.4f}%)")

    def _hook_qkv(self, module, input, output):
        self._qkv_out = output

    def forward(self, x: torch.Tensor):
        b, _, h, w = x.shape

        h_new = (h // self.patch_size) * self.patch_size
        w_new = (w // self.patch_size) * self.patch_size
        x = CenterCrop((h_new, w_new))(x)

        x = self.dino.prepare_tokens_with_masks(x)

        # frozen prefix (for consistency with current project strategy)
        with torch.no_grad():
            for blk in self.dino.blocks[: self.split]:
                x = blk(x)

        x = x.detach()

        # trainable tail
        for blk in self.dino.blocks[self.split : self.num_layers]:
            x = blk(x)

        x = self.dino.norm(x)

        cls_token = x[:, 0]

        ph, pw = h_new // self.patch_size, w_new // self.patch_size
        n_patches = ph * pw
        patch_tokens = x[:, -n_patches:, :]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(b, self.out_channels, ph, pw)

        attn = None
        if self._qkv_out is not None:
            qkv = self._qkv_out
            self._qkv_out = None
            b_tok, n_tok, _ = qkv.shape

            last_blk = self.dino.blocks[self.num_layers - 1]
            num_heads = last_blk.attn.num_heads
            head_dim = self.out_channels // num_heads
            scale = head_dim ** -0.5

            qkv = qkv.reshape(b_tok, n_tok, 3, num_heads, head_dim)
            q, k, _ = qkv.unbind(2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)

        return cls_token, patch_tokens, attn
