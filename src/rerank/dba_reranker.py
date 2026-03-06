"""Database augmentation (DBA) reranker."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .rerankers import register
from .faiss_utils import faiss_knn_topk, aggregate_neighbors

DBA_K = 30  # recommended: 10-20
# Softmax temperature for DBA neighbor weighting
DBA_ALPHA = 3  # recommended: 2.0-3.0
# Weight of the aggregated neighbor vector to add to gallery features
DBA_LAMBDA = 0.3 # recommended: 0.1-0.3


@register("dba")
def dba_reranker(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    *,
    k: int = DBA_K,
    alpha: float = DBA_ALPHA,
    gamma: float = DBA_LAMBDA,
    metric: str = "cosine",
    prefer_gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Database Augmentation (DBA) reranker.

    Args:
        q_feats: Query feature tensor (unused by DBA) of shape (Nq, D).
        g_feats: Gallery feature tensor of shape (Ng, D).
        k: Number of neighbors used for DBA (search k + self).
        alpha: Softmax temperature used when aggregating neighbor features.
        gamma: Weight applied to the aggregated neighbor vector when augmenting gallery.
        metric: Distance metric for FAISS search.
        prefer_gpu: Whether to prefer FAISS GPU index when available.

    Returns:
        q_emb: Unchanged query embeddings (Nq, D).
        g_emb: Augmented gallery embeddings (Ng, D).

    Examples:
        >>> q_emb, g_emb = dba_reranker(q_feats, g_feats, k=20)
    """

    g = F.normalize(g_feats.float(), p=2, dim=1)

    sim, idx = faiss_knn_topk(
        g,
        k=min(k + 1, g.shape[0]),
        metric=metric,
        database=g,
        prefer_gpu=prefer_gpu,
    )

    idx = idx[:, 1:]
    sim = sim[:, 1:]

    expanded = aggregate_neighbors(g, idx, sim, alpha=alpha)
    g_emb = F.normalize(g + gamma * expanded, p=2, dim=1)

    return q_feats, g_emb
