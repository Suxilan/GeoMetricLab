import torch
import torch.nn as nn
from typing import Optional


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GridCrossBlock(nn.Module):
    """Alternating grid-then-cross attention block.

    - Grid attention: per-image attention across N grid tokens.
    - Cross attention: per-token attention across B images.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float, ffn_dim: int = 2048):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)

        # self.norm_grid = nn.LayerNorm(self.dim)
        # self.attn_grid = nn.MultiheadAttention(self.dim, self.num_heads, dropout=dropout, batch_first=False)

        self.norm_cross = nn.LayerNorm(self.dim)
        self.attn_cross = nn.MultiheadAttention(self.dim, self.num_heads, dropout=dropout, batch_first=False)
        self.norm_ffn = nn.LayerNorm(self.dim)
        self.ffn = _FeedForward(self.dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, D)
        # 1) grid self-attn (post-norm): seq=N, batch=B
        # h = x.transpose(0, 1)  # (N, B, D)
        # h, _ = self.attn_grid(h, h, h, need_weights=False)
        # x = x + h.transpose(0, 1)  # (B, N, D)
        # x = self.norm_grid(x)

        # 2) cross-image attn (post-norm): seq=B, batch=N
        h, _ = self.attn_cross(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm_cross(x + h)
        x = self.norm_ffn(x + self.ffn(x))
        return x


class GCE(nn.Module):
    """Geometric Consensus Encoder (GCE).

    Alternates grid self-attention and cross-image attention.
    Accepts (B, D) or (B, N, D) and returns the same shape.
    """

    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        requested_heads = int(num_heads)
        self.num_heads = requested_heads if self.dim % requested_heads == 0 else 1
        self.encoder = nn.ModuleList([
            _GridCrossBlock(self.dim, self.num_heads, dropout, ffn_dim=2048)
            for _ in range(2)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, D) or (B, N, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"x must be (B, D) or (B, N, D), got {tuple(x.shape)}")

        out = x
        for layer in self.encoder:
            out = layer(out, attn_mask=attn_mask)
        return out.squeeze(1) if out.size(1) == 1 else out
