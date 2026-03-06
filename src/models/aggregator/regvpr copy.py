import torch
import torch.nn as nn
from typing import Tuple, Optional, Any, Sequence

from .base import AggregatorBase
from ..modules.attention import RoPEAttention


class _RoPEBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, hidden: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=True,
            qkv_bias=True,
            proj_bias=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        prefix_tokens: int,
        patch_hw: Optional[Tuple[int, int]] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), prefix_tokens=prefix_tokens, patch_hw=patch_hw, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class RegVPR(AggregatorBase):
    """Register-token Transformer aggregator for VPR.

    Input: local feature map/tokens from backbone + external CLS token.
    - Create learnable register tokens.
    - Concatenate [CLS, registers, patches] and run RoPE Transformer blocks.
    - RoPE is applied only on patch tokens (skip CLS and registers).
    - Keep final CLS token as descriptor.

    Output descriptor dim: in_channels
    """

    def __init__(
        self,
        in_channels: int,
        num_register_tokens: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        enable_cross_frame: bool = True,
        aa_order: Sequence[str] = ("frame", "global"),
        aa_block_size: int = 1,
    ):
        super().__init__(in_channels=in_channels)
        if in_channels is None:
            raise ValueError("RegVPRAggregator requires in_channels")
        if num_register_tokens <= 0:
            raise ValueError("num_register_tokens must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if in_channels % 8 != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by num_heads ({8})")

        self.num_register_tokens = int(num_register_tokens)
        self.out_channels = in_channels
        self.enable_cross_frame = bool(enable_cross_frame)
        self.aa_order = list(aa_order)
        self.aa_block_size = int(aa_block_size)
        if self.aa_block_size <= 0:
            raise ValueError("aa_block_size must be > 0")

        self.register_tokens = nn.Parameter(torch.randn(1, self.num_register_tokens, in_channels) * 0.02)

        if not self.enable_cross_frame:
            # Default mode: keep original behavior (independent per-frame processing).
            self.blocks = nn.ModuleList([
                _RoPEBlock(dim=in_channels, num_heads=8, dropout=dropout) for _ in range(num_layers)
            ])
            self.frame_blocks = None
            self.global_blocks = None
            self.aa_block_num = 0
        else:
            # Cross-frame mode: alternating frame/global attention.
            if num_layers % self.aa_block_size != 0:
                raise ValueError(
                    f"num_layers ({num_layers}) must be divisible by aa_block_size ({self.aa_block_size})"
                )
            self.frame_blocks = nn.ModuleList([
                _RoPEBlock(dim=in_channels, num_heads=8, dropout=dropout) for _ in range(num_layers)
            ])
            self.global_blocks = nn.ModuleList([
                _RoPEBlock(dim=in_channels, num_heads=8, dropout=dropout) for _ in range(num_layers)
            ])
            self.blocks = None
            self.aa_block_num = num_layers // self.aa_block_size

        self.norm = nn.LayerNorm(in_channels)

    def _apply_global_block(
        self,
        seq: torch.Tensor,
        blk: _RoPEBlock,
        global_rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Global attention across B dimension in VGGT-style order.

        Input `seq`: (B, P, C), where P=(special + patch) tokens per frame.
        Flatten by frame order to (1, B*P, C), run one global block, then reshape back.
        """
        B, P, C = seq.shape
        global_seq = seq.reshape(1, B * P, C)
        # We pass full-length rope (including special-token identity entries), so no prefix skip here.
        global_seq = blk(global_seq, prefix_tokens=0, rope=global_rope)
        return global_seq.reshape(B, P, C)

    def forward(
        self,
        x: torch.Tensor,
        cls_token: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if cls_token is None:
            raise ValueError("RegVPRAggregator requires cls_token (from ViT backbone).")

        if cls_token.dim() == 3 and cls_token.shape[1] == 1:
            cls_token = cls_token[:, 0, :]

        if x.dim() == 4:
            # (B, C, H, W) -> (B, N, C)
            b, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            frame_patch_hw = (h, w)
        elif x.dim() == 3:
            # (B, N, C)
            b, n, c = x.shape
            tokens = x
            side = int(n ** 0.5)
            if side * side != n:
                raise ValueError(
                    f"When input is (B, N, C), N must be a square number for default 2D RoPE. Got N={n}."
                )
            frame_patch_hw = (side, side)
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {x.dim()}D")

        if cls_token.dim() != 2:
            raise ValueError(f"cls_token must be (B, C), got {tuple(cls_token.shape)}")

        cls = cls_token.unsqueeze(1)  # (B, 1, C)
        reg = self.register_tokens.expand(tokens.size(0), -1, -1)  # (B, R, C)
        seq = torch.cat([cls, reg, tokens], dim=1)  # (B, 1+R+N, C)

        prefix_tokens = 1 + self.num_register_tokens
        rope_builder = self.frame_blocks[0].attn if self.enable_cross_frame else self.blocks[0].attn
        frame_rope = rope_builder.build_rope(
            patch_hw=frame_patch_hw,
            token_repeat=1,
            device=seq.device,
            dtype=torch.float32,
        )

        if not self.enable_cross_frame:
            for blk in self.blocks:
                seq = blk(seq, prefix_tokens=prefix_tokens, rope=frame_rope)
        else:
            frame_idx = 0
            global_idx = 0
            # Build full-length global rope in frame-major order:
            # per frame: [special(identity), patch(2D rope)], then flatten over B.
            frame_sin, frame_cos = frame_rope  # (1,1,N_patch,D_head)
            n_patch = frame_sin.shape[-2]
            d_head = frame_sin.shape[-1]

            special_sin = torch.zeros(prefix_tokens, d_head, device=seq.device, dtype=torch.float32)
            special_cos = torch.ones(prefix_tokens, d_head, device=seq.device, dtype=torch.float32)

            frame_sin_2d = frame_sin[0, 0]  # (N_patch, D_head)
            frame_cos_2d = frame_cos[0, 0]  # (N_patch, D_head)

            one_frame_sin = torch.cat([special_sin, frame_sin_2d], dim=0)  # (P, D_head)
            one_frame_cos = torch.cat([special_cos, frame_cos_2d], dim=0)  # (P, D_head)

            B_frames = seq.shape[0]
            global_sin = one_frame_sin.unsqueeze(0).expand(B_frames, -1, -1).reshape(B_frames * (prefix_tokens + n_patch), d_head)
            global_cos = one_frame_cos.unsqueeze(0).expand(B_frames, -1, -1).reshape(B_frames * (prefix_tokens + n_patch), d_head)
            global_rope = (global_sin.unsqueeze(0).unsqueeze(0), global_cos.unsqueeze(0).unsqueeze(0))

            for _ in range(self.aa_block_num):
                for attn_type in self.aa_order:
                    if attn_type == "frame":
                        for _ in range(self.aa_block_size):
                            seq = self.frame_blocks[frame_idx](
                                seq,
                                prefix_tokens=prefix_tokens,
                                rope=frame_rope,
                            )
                            frame_idx += 1
                    elif attn_type == "global":
                        for _ in range(self.aa_block_size):
                            seq = self._apply_global_block(
                                seq,
                                blk=self.global_blocks[global_idx],
                                global_rope=global_rope,
                            )
                            global_idx += 1
                    else:
                        raise ValueError(f"Unknown attn_type in aa_order: {attn_type}")

        seq = self.norm(seq)
        cls_out = seq[:, 0, :]
        return cls_out, None
