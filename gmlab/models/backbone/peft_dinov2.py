"""DINOv2 LoRA backbone with an explicit local repository and feature contract."""

import logging
from pathlib import Path
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import functional as TF


class PEFTDinoV2(nn.Module):
    """Return (CLS [N,C], patches [N,C,H/14,W/14], attention or None).

    Images are center-cropped to patch-size multiples, as in the original
    backbone. Only LoRA in the selected tail and final norm are trainable;
    the prefix runs under no_grad and its output is detached. A local DINOv2
    repository is required unless an ordinary DINO module is injected.

    Attention reconstruction is optional because SCPP only consumes patches.
    Turning it off changes neither patch descriptors nor trainable parameters.
    gradient_checkpointing trades recomputation for activation memory in train
    mode using native nonreentrant PyTorch checkpointing with RNG preservation.
    """

    def __init__(
        self,
        backbone_name: str = "peft_dinov2_vitb14",
        num_unfrozen_blocks: int = 1,
        num_layers: int | None = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_bias: str = "none",
        lora_targets: Sequence[str] = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
        *,
        repository: str | Path = "third_party/dinov2",
        pretrained: bool = True,
        capture_attention: bool = False,
        gradient_checkpointing: bool = False,
        dino: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if capture_attention and gradient_checkpointing:
            raise ValueError(
                "Attention capture retains activations; choose it or gradient checkpointing"
            )
        if lora_bias != "none":
            raise ValueError(
                "This backbone trains LoRA and final norm; lora_bias must be 'none'"
            )
        if not backbone_name.startswith("peft_dinov2_"):
            raise ValueError("Expected a peft_dinov2_* backbone name")
        self.backbone_name = backbone_name
        self.base_backbone_name = backbone_name.removeprefix("peft_")
        if dino is None:
            repository = Path(repository).resolve()
            if not (repository / "hubconf.py").is_file():
                raise FileNotFoundError(f"DINOv2 hubconf.py not found in {repository}")
            dino = torch.hub.load(
                str(repository),
                self.base_backbone_name,
                source="local",
                pretrained=pretrained,
            )
        self.dino = dino
        self.total_layers = len(dino.blocks)
        self.num_layers = self.total_layers if num_layers is None else num_layers
        if not 1 <= self.num_layers <= self.total_layers:
            raise ValueError("num_layers must select a nonempty prefix of DINO blocks")
        if not 1 <= num_unfrozen_blocks <= self.num_layers:
            raise ValueError("num_unfrozen_blocks must be in [1, num_layers]")
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.split = self.num_layers - num_unfrozen_blocks
        self.patch_size = int(dino.patch_size)
        self.out_channels = int(dino.embed_dim)
        self.num_register_tokens = int(dino.num_register_tokens)
        self.capture_attention = capture_attention
        self.gradient_checkpointing = gradient_checkpointing
        self._qkv_out = None

        for parameter in self.dino.parameters():
            parameter.requires_grad_(False)
        self._inject_lora(lora_r, lora_alpha, lora_dropout, lora_bias, lora_targets)
        for parameter in self.dino.norm.parameters():
            parameter.requires_grad_(True)
        if capture_attention:
            self.dino.blocks[self.num_layers - 1].attn.qkv.register_forward_hook(
                self._capture_qkv
            )
        logging.getLogger(__name__).info(
            "DINOv2 LoRA: %s, active blocks [%d,%d), attention=%s",
            self.base_backbone_name,
            self.split,
            self.num_layers,
            capture_attention,
        )

    def _inject_lora(
        self, rank: int, alpha: int, dropout: float, bias: str, targets: Sequence[str]
    ) -> None:
        from peft import LoraConfig, get_peft_model

        prefixes = tuple(
            f"blocks.{index}." for index in range(self.split, self.num_layers)
        )
        selected = sorted(
            name
            for name, module in self.dino.named_modules()
            if isinstance(module, nn.Linear)
            and name.startswith(prefixes)
            and any(name.endswith(target) for target in targets)
        )
        if not selected:
            raise ValueError(f"No LoRA Linear modules match targets {tuple(targets)}")
        # PEFT injects into the base model in place. Keep its native token/block
        # interface and state names rather than registering a second wrapper.
        get_peft_model(
            self.dino,
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=bias,
                target_modules=selected,
            ),
        )
        for name, parameter in self.dino.named_parameters():
            parameter.requires_grad_("lora_" in name)

    def _capture_qkv(
        self, module: nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        self._qkv_out = output

    def _attention(self) -> torch.Tensor | None:
        qkv = self._qkv_out
        self._qkv_out = None
        if qkv is None:
            return None
        batch_size, token_count, _ = qkv.shape
        head_count = self.dino.blocks[self.num_layers - 1].attn.num_heads
        head_dim = self.out_channels // head_count
        query, key, _ = qkv.reshape(
            batch_size, token_count, 3, head_count, head_dim
        ).unbind(2)
        scores = query.transpose(1, 2) @ key.transpose(1, 2).transpose(-2, -1)
        return (scores * head_dim**-0.5).softmax(dim=-1)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, _, height, width = images.shape
        crop_height = height // self.patch_size * self.patch_size
        crop_width = width // self.patch_size * self.patch_size
        if min(crop_height, crop_width) == 0:
            raise ValueError("Input image must contain at least one DINO patch")
        tokens = self.dino.prepare_tokens_with_masks(
            TF.center_crop(images, [crop_height, crop_width])
        )
        with torch.no_grad():
            for block in self.dino.blocks[: self.split]:
                tokens = block(tokens)
        tokens = tokens.detach()
        for block in self.dino.blocks[self.split : self.num_layers]:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.dino.norm(tokens)
        patch_height, patch_width = (
            crop_height // self.patch_size,
            crop_width // self.patch_size,
        )
        patches = tokens[:, -(patch_height * patch_width) :, :]
        patches = patches.transpose(1, 2).reshape(
            batch_size, self.out_channels, patch_height, patch_width
        )
        return tokens[:, 0], patches, self._attention()
