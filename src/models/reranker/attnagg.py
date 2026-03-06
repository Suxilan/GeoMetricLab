from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class RoPE1D(nn.Module):
	def __init__(self, *, head_dim: int, base: float = 10000.0):
		super().__init__()
		if head_dim % 2 != 0:
			raise ValueError(f"head_dim must be even for RoPE (got {head_dim})")
		inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
		self.register_buffer("inv_freq", inv_freq, persistent=False)

	def forward(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
		t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
		freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (L, Dh/2)
		angles = torch.cat([freqs, freqs], dim=-1).to(dtype=dtype)  # (L, Dh)
		return angles.sin(), angles.cos()


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
	x1, x2 = x.chunk(2, dim=-1)
	return torch.cat((-x2, x1), dim=-1)


def _rope_apply(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
	sin = sin[None, None, :, :]
	cos = cos[None, None, :, :]
	return (x * cos) + (_rope_rotate_half(x) * sin)


class SelfAttention(nn.Module):
	def __init__(
		self,
		*,
		dim: int,
		num_heads: int = 8,
		qkv_bias: bool = True,
		proj_bias: bool = True,
		attn_drop: float = 0.0,
		proj_drop: float = 0.0,
		rope_base: float = 10000.0,
	) -> None:
		super().__init__()
		if dim % num_heads != 0:
			raise ValueError("dim must be divisible by num_heads")
		self.num_heads = int(num_heads)
		self.dim = int(dim)
		self.head_dim = self.dim // self.num_heads

		self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=qkv_bias)
		self.proj = nn.Linear(self.dim, self.dim, bias=proj_bias)
		self.attn_drop = float(attn_drop)
		self.proj_drop = nn.Dropout(proj_drop)
		self.rope = RoPE1D(head_dim=self.head_dim, base=rope_base)

	def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
		q_dtype = q.dtype
		k_dtype = k.dtype
		sin, cos = rope
		rope_dtype = sin.dtype
		q = q.to(dtype=rope_dtype)
		k = k.to(dtype=rope_dtype)
		q = _rope_apply(q, sin, cos)
		k = _rope_apply(k, sin, cos)
		return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

	def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		bsz, seq_len, _ = x.shape
		qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
		q, k, v = torch.unbind(qkv, dim=2)
		q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B,H,L,Dh)

		rope = self.rope(seq_len, device=x.device, dtype=torch.float32)
		q, k = self._apply_rope(q, k, rope)

		attn_mask = None
		if key_padding_mask is not None:
			# (B, L) -> (B, 1, 1, L), True means masked
			attn_mask = key_padding_mask[:, None, None, :]

		out = F.scaled_dot_product_attention(
			q,
			k,
			v,
			attn_mask=attn_mask,
			dropout_p=self.attn_drop if self.training else 0.0,
			is_causal=False,
		)
		out = out.transpose(1, 2).reshape(bsz, seq_len, self.dim)
		return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
	def __init__(
		self,
		*,
		embed_dim: int,
		num_heads: int,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
		rope_base: float = 10000.0,
	):
		super().__init__()
		self.norm1 = nn.LayerNorm(embed_dim)
		self.attn = SelfAttention(
			dim=embed_dim,
			num_heads=num_heads,
			qkv_bias=True,
			proj_bias=True,
			attn_drop=dropout,
			proj_drop=dropout,
			rope_base=rope_base,
		)
		self.norm2 = nn.LayerNorm(embed_dim)

		hidden = int(embed_dim * mlp_ratio)
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, embed_dim),
			nn.Dropout(dropout),
		)

	def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
		x = x + self.mlp(self.norm2(x))
		return x


class AttnAgg(nn.Module):
	def __init__(
		self,
		*,
		embed_dim: int = 512,
		num_heads: int = 8,
		num_layers: int = 2,
		mlp_ratio: float = 4.0,
		dropout: float = 0.0,
		rope_base: float = 10000.0,
	):
		super().__init__()
		self.embed_dim = int(embed_dim)

		self.blocks = nn.ModuleList(
			[
				TransformerBlock(
					embed_dim=self.embed_dim,
					num_heads=num_heads,
					mlp_ratio=mlp_ratio,
					dropout=dropout,
					rope_base=rope_base,
				)
				for _ in range(int(num_layers))
			]
		)
		self.norm = nn.LayerNorm(self.embed_dim)
		self.aux_classifier = nn.Linear(self.embed_dim, 1)
		nn.init.xavier_uniform_(self.aux_classifier.weight)
		if self.aux_classifier.bias is not None:
			nn.init.zeros_(self.aux_classifier.bias)

	def forward(
		self,
		seq: torch.Tensor,
		*,
		key_padding_mask: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		if seq.ndim != 3:
			raise ValueError(f"Expected (B, L, D) got {tuple(seq.shape)}")
		if seq.shape[-1] != self.embed_dim:
			raise ValueError(f"Expected last dim {self.embed_dim}, got {int(seq.shape[-1])}")

		x = seq
		for blk in self.blocks:
			x = blk(x, key_padding_mask=key_padding_mask)
		
		q = x[:, 0, :]  # (B, D)
		d = x[:, 1:, :]  # (B, L-1, D)

		q = q + seq[:, 0, :]
		q = self.norm(q)

		# Auxiliary logits for token-level supervision
		aux_logits = self.aux_classifier(d).squeeze(-1)  # (B, L-1)

		return F.normalize(q, p=2, dim=-1), aux_logits

