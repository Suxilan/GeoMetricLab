"""Average Query Expansion reranker."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .neighbors import aggregate_neighbors, knn_topk
from .rerankers import register

AQE_K = 100  # recommended: 5-10
# Strength of softmax when weighting neighbors during aggregation.
AQE_ALPHA = 10  # recommended: 3.0-4.0
# Weight of the aggregated neighbor vector when expanding the query.
AQE_LAMBDA = 1.0  # recommended: 0.2-0.6


@register("aqe")
def aqe_reranker(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    *,
    k: int = AQE_K,
    alpha: float = AQE_ALPHA,
    gamma: float = AQE_LAMBDA,
    metric: str = "cosine",
    prefer_gpu: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average Query Expansion (AQE) reranker.

    Args:
        q_feats: Query feature tensor of shape (Nq, D), float.
        g_feats: Gallery feature tensor of shape (Ng, D), float.
        k: Number of neighbors to use for averaging (search k).
        alpha: Softmax temperature for neighbor weighting.
        gamma: Expansion multiplier controlling strength of added neighbor vector.
        metric: Distance metric used for neighbor search ("cosine" or "l2").
        prefer_gpu: Whether to prefer FAISS GPU index when available.

    Returns:
        q_emb: Expanded query embeddings (Nq, D).
        g_emb: Unchanged gallery embeddings (Ng, D).

    Examples:
        >>> q_exp, g_exp = aqe_reranker(q_feats, g_feats, k=5)
    """

    q = F.normalize(q_feats.float(), p=2, dim=1)
    g = F.normalize(g_feats.float(), p=2, dim=1)

    sim, idx = knn_topk(
        q,
        k=min(k, g.shape[0]),
        metric=metric,
        database=g,
        prefer_gpu=prefer_gpu,
    )

    expanded = aggregate_neighbors(g, idx, sim, alpha=alpha)
    q_emb = F.normalize(q + gamma * expanded, p=2, dim=1)

    return q_emb, g_feats
