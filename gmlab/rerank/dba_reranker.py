"""Database augmentation (DBA) reranker."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .neighbors import aggregate_neighbors, knn_topk
from .rerankers import register

DBA_K = 30  # recommended: 10-20
# Softmax temperature for DBA neighbor weighting
DBA_ALPHA = 3  # recommended: 2.0-3.0
# Weight of the aggregated neighbor vector to add to gallery features
DBA_LAMBDA = 0.3  # recommended: 0.1-0.3


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Database Augmentation (DBA) reranker.

    Args:
        q_feats: Query feature tensor (unused by DBA) of shape (Nq, D).
        g_feats: Gallery feature tensor of shape (Ng, D).
        k: Number of neighbors used for DBA (search k + self).
        alpha: Softmax temperature used when aggregating neighbor features.
        gamma: Weight applied to the aggregated neighbor vector when augmenting gallery.
        metric: Distance metric for neighbor search.
        prefer_gpu: Whether to prefer FAISS GPU index when available.

    Returns:
        q_emb: Unchanged query embeddings (Nq, D).
        g_emb: Augmented gallery embeddings (Ng, D).

    Examples:
        >>> q_emb, g_emb = dba_reranker(q_feats, g_feats, k=20)
    """

    g = F.normalize(g_feats.float(), p=2, dim=1)

    sim, idx = knn_topk(
        g,
        k=min(k + 1, g.shape[0]),
        metric=metric,
        database=g,
        prefer_gpu=prefer_gpu,
    )

    # Equal descriptors need not put self first. Remove it by row identity.
    self_mask = idx == torch.arange(len(g))[:, None]
    order = self_mask.to(torch.int64).argsort(dim=1, stable=True)
    keep = order[:, : min(k, len(g) - 1)]
    idx = idx.gather(1, keep)
    sim = sim.gather(1, keep)

    expanded = aggregate_neighbors(g, idx, sim, alpha=alpha)
    g_emb = F.normalize(g + gamma * expanded, p=2, dim=1)

    return q_feats, g_emb
