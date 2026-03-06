"""k-Reciprocal Nearest Neighbors (k-RNN) reranking implementation.

This module implements a vectorized k-reciprocal neighbor strategy with
soft-weight aggregation. The implementation preserves the original algorithmic
steps but standardizes the function interface and docstrings.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .rerankers import register
from .faiss_utils import faiss_knn_topk, aggregate_neighbors

KRNN_K1 = 40  # recommended: 40-60 (initial top-k search)
KRNN_K2 = 15  # recommended: 5-15 (reciprocal/truncation window)
KRNN_ALPHA = 0.5  # controls exp(-dist) weighting after converting sim->dist
KRNN_LAMBDA = 1.0  # residual fusion strength


def get_k_reciprocal_mask(rank: torch.Tensor, k: int) -> torch.Tensor:
    """Compute reciprocal mask for the first ``k`` neighbors.

    Args:
        rank: Integer neighbor indices of shape (N, Kmax) where each row lists
            the top-K neighbors returned by FAISS (including self at position 0).
        k: Number of neighbors to consider (Kmax may be larger).

    Returns:
        A boolean tensor of shape (N, k) where True indicates the forward
        neighbor is reciprocal (i.e., target appears in neighbor's top-k).
    """
    N = rank.size(0)
    device = rank.device

    forward = rank[:, :k]
    target = torch.arange(N, device=device).view(N, 1)
    mask = torch.empty((N, k), device=device, dtype=torch.bool)

    for j in range(k):
        nb = forward[:, j]
        back = rank.index_select(0, nb)[:, :k]
        mask[:, j] = (back == target).any(dim=1)

    return mask


@torch.no_grad()
@register("krnn")
def krnn_reranker(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    *,
    k1: int = KRNN_K1,
    k2: int = KRNN_K2,
    alpha: float = KRNN_ALPHA,
    lambd: float = KRNN_LAMBDA,
    metric: str = "cosine",
    prefer_gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """k-RNN reranker with soft aggregation and residual fusion.

    Args:
        q_feats: Query features tensor of shape (Nq, D).
        g_feats: Gallery features tensor of shape (Ng, D).
        k1: Number of neighbors to retrieve for building reciprocal sets (search k).
        k2: Truncation size / local expansion for stage-2 aggregation.
        alpha: Exponential weighting constant applied to (1 - sim) distance.
        lambd: Residual weight between original features and aggregated features.
        metric: FAISS distance metric ("cosine" or "l2").
        prefer_gpu: Whether to prefer FAISS GPU index when available.

    Returns:
        q_emb: Refined query embeddings (Nq, D).
        g_emb: Refined gallery embeddings (Ng, D).

    Examples:
        >>> q_emb, g_emb = krnn_reranker(q_feats, g_feats, k1=50, k2=10)
    """

    q = F.normalize(q_feats.float(), p=2, dim=1)
    g = F.normalize(g_feats.float(), p=2, dim=1)
    features = torch.cat((q, g), dim=0)
    N = features.size(0)

    k1_search = min(k1 + 1, N)
    k2_eff = min(k2, k1)

    sims, idx = faiss_knn_topk(features, k=k1_search, metric=metric, prefer_gpu=prefer_gpu)

    device = features.device
    sims = sims.to(device, dtype=torch.float32, non_blocking=True)
    idx = idx.to(device, dtype=torch.long, non_blocking=True)

    # ensure self is at position 0
    self_ids = torch.arange(N, device=device)
    if not (idx[:, 0] == self_ids).all():
        idx[:, 0] = self_ids
        sims[:, 0] = 1.0

    mask = get_k_reciprocal_mask(idx, k1_search)

    logits = sims - 1.0
    logits.masked_fill_(~mask, float("-inf"))

    feats_stage1 = aggregate_neighbors(
        features,
        idx,
        logits,
        alpha=alpha,
    )

    if k2_eff > 1:
        lqe_idx = idx[:, :k2_eff].flatten()
        feats_stage2 = feats_stage1.index_select(0, lqe_idx).view(N, k2_eff, -1).mean(dim=1)
    else:
        feats_stage2 = feats_stage1

    features_final = F.normalize(features + lambd * feats_stage2, p=2, dim=1)

    q_emb = features_final[: q.size(0)]
    g_emb = features_final[q.size(0) :]
    return q_emb, g_emb