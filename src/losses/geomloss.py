"""GeoMetricLoss: wrapper around pytorch-metric-learning losses/miners."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import print_rank_0
from losses.config import (
    LOSS_REGISTRY,
    LOSS_FACTORIES,
    MINER_FACTORIES,
    MINER_OUTPUT_SHAPE,
)


class GeoMetricLoss(nn.Module):
    """Wrap common metric-learning losses and optional miners.

    Args:
        loss_name: Name of the loss (e.g., "triplet", "contrastive", "multisimilarity", "instance").
        loss_params: Keyword arguments for the loss constructor.
        miner_name: Optional miner name (e.g., "batch_hard", "multi_similarity").
        miner_params: Keyword arguments for the miner constructor.
        normalize: If True, L2-normalize embeddings before loss.
    """

    def __init__(
        self,
        loss_name: str,
        loss_params: Optional[Dict[str, Any]] = None,
        miner_name: Optional[str] = None,
        miner_params: Optional[Dict[str, Any]] = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.loss_name = (loss_name or "").lower()
        self.miner_name = (miner_name or "").lower() if miner_name else None
        self.normalize = normalize

        if loss_name not in LOSS_FACTORIES:
            raise ValueError(f"Unsupported loss: {loss_name}. Supported: {list(LOSS_FACTORIES)}")

        if loss_name not in LOSS_REGISTRY:
            raise ValueError(f"Loss '{loss_name}' missing from LOSS_REGISTRY whitelist")

        # --- Validation ---
        if self.loss_name not in LOSS_REGISTRY:
            raise ValueError(f"Unsupported loss: {self.loss_name}. Supported: {list(LOSS_REGISTRY.keys())}")
        
        self.config = LOSS_REGISTRY[self.loss_name]
        
        if self.miner_name:
            if self.miner_name not in MINER_FACTORIES:
                raise ValueError(f"Unsupported miner: {self.miner_name}")
            
            if self.miner_name not in self.config.compatible_miners:
                raise ValueError(f"Miner '{self.miner_name}' incompatible with loss '{self.loss_name}'")
            
            out_shape = MINER_OUTPUT_SHAPE.get(self.miner_name)
            if out_shape != self.config.tuple_len:
                raise ValueError(f"Miner output shape {out_shape} != Loss input shape {self.config.tuple_len}")

        # --- Initialization ---
        self.loss_fn = LOSS_FACTORIES[self.loss_name](**(loss_params or {}))
        self.miner = MINER_FACTORIES[self.miner_name](**(miner_params or {})) if self.miner_name else None

        print_rank_0(f"[{self.__class__.__name__}] Initialized with loss='{self.loss_name}', miner='{self.miner_name}'")
        print_rank_0(f"[{self.__class__.__name__}]    Loss params: {loss_params}")
        print_rank_0(f"[{self.__class__.__name__}]    Miner params: {miner_params}")
        print_rank_0(f"[{self.__class__.__name__}]    Loss Normalize embeddings: {self.normalize}")

    def forward(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        indices_tuple: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        
        if self.normalize:
            feats = F.normalize(feats, p=2, dim=1)

        final_indices = None
        stats = {}

        # --- Logic Branching ---
        
        # 1. Intersection Mode: Explicit Geometry + Miner
        if indices_tuple is not None and self.miner is not None:
            self._check_tuple_shape(indices_tuple)
            
            # A. Mine hard samples using labels
            miner_indices = self.miner(feats, labels)
            
            # B. Filter using geometry whitelist
            final_indices = _intersect_mining(miner_indices, indices_tuple, feats.shape[0])
            
            # C. Log keep rate
            orig = max(miner_indices[0].numel(), 1)
            keep = final_indices[0].numel() if final_indices else 0
            stats["intersection_keep_rate"] = torch.tensor(
                keep / orig, dtype=torch.float32, device=feats.device
            )

        # 2. Static Mode: Explicit Geometry Only
        elif indices_tuple is not None:
            self._check_tuple_shape(indices_tuple)
            if not self.config.allow_explicit_indices:
                raise ValueError(f"Loss '{self.loss_name}' does not allow explicit indices")
            final_indices = indices_tuple

        # 3. Miner Mode: Implicit Mining (Class-based)
        elif self.miner is not None:
            final_indices = self.miner(feats, labels)
            self._check_tuple_shape(final_indices, is_miner_out=True)

        # --- Loss Computation ---
        # Note: If final_indices is None (University mode w/o miner), PML uses labels directly.
        if final_indices is not None:
            loss_val = self.loss_fn(feats, None, indices_tuple=final_indices)
        else:
            loss_val = self.loss_fn(feats, labels)

        # --- Stats ---
        out = {"loss": loss_val, f"loss_{self.loss_name}": loss_val}
        
        if final_indices is not None:
            pos_frac, neg_frac = _compute_miner_fraction(final_indices, device=feats.device)
            out["active_pos_frac"] = pos_frac
            out["active_neg_frac"] = neg_frac
            
        out.update(stats)
        return out

    def _check_tuple_shape(self, t: tuple, is_miner_out: bool = False):
        if len(t) != self.config.tuple_len:
            src = f"Miner '{self.miner_name}'" if is_miner_out else "Input indices_tuple"
            raise ValueError(f"{src} length {len(t)} mismatch. Loss '{self.loss_name}' expects {self.config.tuple_len}")

def _intersect_mining(miner_out: tuple, geo_out: tuple, batch_size: int) -> tuple:
    """Keep miner pairs only if they exist in the geometry whitelist."""
    device = miner_out[0].device
    
    # Create masks from geometry indices
    valid_pos = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
    valid_neg = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)

    # Triplet: (a, p, n)
    if len(miner_out) == 3:
        m_a, m_p, m_n = miner_out
        g_a, g_p, g_n = geo_out
        
        valid_pos[g_a, g_p] = True
        valid_neg[g_a, g_n] = True
        
        keep = valid_pos[m_a, m_p] & valid_neg[m_a, m_n]
        if keep.sum() == 0: return tuple(torch.empty(0, dtype=torch.long, device=device) for _ in range(3))
        return (m_a[keep], m_p[keep], m_n[keep])

    # Pair: (a1, p, a2, n)
    elif len(miner_out) == 4:
        m_a1, m_p, m_a2, m_n = miner_out
        g_a1, g_p, g_a2, g_n = geo_out

        valid_pos[g_a1, g_p] = True
        valid_neg[g_a2, g_n] = True

        keep_pos = valid_pos[m_a1, m_p]
        keep_neg = valid_neg[m_a2, m_n]

        if keep_pos.sum() == 0 and keep_neg.sum() == 0:
            return tuple(torch.empty(0, dtype=torch.long, device=device) for _ in range(4))

        return (m_a1[keep_pos], m_p[keep_pos], m_a2[keep_neg], m_n[keep_neg])

    return miner_out

def _compute_miner_fraction(indices: Tuple[torch.Tensor, ...], device: torch.device | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate positive/negative pair coverage from miner output.

    Supports standard PML tuple miners returning (a1, p, a2, n) or (a, p, n).
    Ensures the returned tensors live on the provided device to avoid CPU all-reduce under DDP.
    """
    if not indices:
        dev = device or torch.device("cpu")
        return torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)

    dev = device or indices[0].device
    p_len = indices[1].numel()
    n_len = indices[2].numel() if len(indices) == 3 else indices[3].numel()
    total = max(p_len + n_len, 1)
    return torch.tensor(p_len / total, device=dev), torch.tensor(n_len / total, device=dev)
