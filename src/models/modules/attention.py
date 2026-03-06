from typing import Optional, Tuple

import torch
import torch.nn as nn

from .rope_position_encoding import RopePositionEmbedding


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Dinov3-style: split last dim into two halves then rotate.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rope_rotate_half(x) * sin)


class RoPEAttention(nn.Module):
    """RoPE multi-head attention using PyTorch's scaled_dot_product_attention.

    Core implementation follows patterns used in Dinov3-style modules:
    - Project Q/K/V
    - Apply RoPE to Q and K
    - Call `torch.nn.functional.scaled_dot_product_attention` for fast attention
    - Project output back to `dim`.

    Args:
        dim: input feature dimension
        num_heads: number of attention heads
        head_dim: per-head dimension (if None, `dim // num_heads` is used)
        dropout: dropout passed to SDPA and output projection
        use_rope: whether to apply rotary embeddings
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            if dim % num_heads != 0:
                raise ValueError("dim must be divisible by num_heads when head_dim is None")
            head_dim = dim // num_heads
        self.head_dim = head_dim
        if self.head_dim % 2 != 0:
            raise ValueError("RoPEAttention requires even head_dim")

        self.qkv = nn.Linear(dim, 3 * num_heads * head_dim, bias=qkv_bias)
        self.proj = nn.Linear(num_heads * head_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_drop = float(dropout)
        self.use_rope = use_rope
        self.rope_base = float(rope_base)
        self.rope = RopePositionEmbedding(
            embed_dim=dim,
            num_heads=num_heads,
            base=self.rope_base,
            dtype=torch.float32,
        )

    @staticmethod
    def _infer_hw(num_patch_tokens: int) -> Tuple[int, int]:
        side = int(num_patch_tokens ** 0.5)
        if side * side != num_patch_tokens:
            raise ValueError(
                f"Cannot infer 2D patch size from {num_patch_tokens} tokens. "
                "Please pass patch_hw=(H, W)."
            )
        return side, side

    def build_rope(
        self,
        patch_hw: Tuple[int, int],
        token_repeat: int = 1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build (sin, cos) for patch tokens using RopePositionEmbedding.

        Returns tensors shaped (1, 1, token_repeat * H*W, head_dim), suitable for
        applying to q/k suffix tokens in attention.
        """
        h_tokens, w_tokens = int(patch_hw[0]), int(patch_hw[1])
        if h_tokens <= 0 or w_tokens <= 0:
            raise ValueError(f"Invalid patch_hw={patch_hw}")

        sin, cos = self.rope(H=h_tokens, W=w_tokens)  # (HW, D_head)
        if token_repeat > 1:
            sin = sin.repeat(token_repeat, 1)
            cos = cos.repeat(token_repeat, 1)

        if device is not None:
            sin = sin.to(device=device, dtype=dtype)
            cos = cos.to(device=device, dtype=dtype)
        else:
            sin = sin.to(dtype=dtype)
            cos = cos.to(dtype=dtype)

        return sin.unsqueeze(0).unsqueeze(0), cos.unsqueeze(0).unsqueeze(0)

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        prefix_tokens: int = 0,
        patch_hw: Optional[Tuple[int, int]] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q/k: (B, H, N, D)
        q_dtype, k_dtype = q.dtype, k.dtype

        seq_len = q.shape[-2] - int(prefix_tokens)
        if seq_len < 0:
            raise ValueError("prefix_tokens cannot be larger than sequence length")
        if seq_len == 0:
            return q, k

        if rope is None:
            if patch_hw is None:
                patch_hw = self._infer_hw(seq_len)
            h_tokens, w_tokens = int(patch_hw[0]), int(patch_hw[1])
            if h_tokens * w_tokens != seq_len:
                raise ValueError(
                    f"patch_hw ({h_tokens}, {w_tokens}) does not match patch token count {seq_len}"
                )
            sin, cos = self.build_rope(
                patch_hw=(h_tokens, w_tokens),
                token_repeat=1,
                device=q.device,
                dtype=torch.float32,
            )
        else:
            sin, cos = rope
            if sin.dim() == 2:
                sin = sin.unsqueeze(0).unsqueeze(0)
            if cos.dim() == 2:
                cos = cos.unsqueeze(0).unsqueeze(0)

        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)

        n = q.shape[-2]
        p = int(prefix_tokens)
        if p < 0 or p > n:
            raise ValueError(f"Invalid prefix_tokens={p} for sequence length {n}")

        if p > 0:
            q_prefix, q_suffix = q[:, :, :p, :], q[:, :, p:, :]
            k_prefix, k_suffix = k[:, :, :p, :], k[:, :, p:, :]
        else:
            q_prefix, q_suffix = None, q
            k_prefix, k_suffix = None, k

        q_suffix = rope_apply(q_suffix, sin, cos)
        k_suffix = rope_apply(k_suffix, sin, cos)

        if p > 0:
            q = torch.cat((q_prefix, q_suffix), dim=-2)
            k = torch.cat((k_prefix, k_suffix), dim=-2)
        else:
            q, k = q_suffix, k_suffix

        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        prefix_tokens: int = 0,
        patch_hw: Optional[Tuple[int, int]] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Self-attention with default 2D RoPE.

        Args:
            x: (B, N, C)
            attn_mask: passed to SDPA
            prefix_tokens: first `prefix_tokens` tokens skip RoPE (e.g. cls/reg)
            patch_hw: optional patch spatial size `(H, W)` for the suffix tokens.
                      If None, try square inference from token count.
            rope: optional precomputed `(sin, cos)` for suffix patch tokens.
        """
        b, n, _ = x.shape

        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        if self.use_rope:
            q, k = self.apply_rope(q, k, prefix_tokens=prefix_tokens, patch_hw=patch_hw, rope=rope)

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        out = self.proj_drop(self.proj(out))
        return out


__all__ = ["RoPEAttention"]
