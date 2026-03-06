"""Diffusion-inspired feature smoothing (DFS) reranker."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .rerankers import register
from .faiss_utils import faiss_knn_topk, aggregate_neighbors_chunked

DFS_K = 30  # recommended: 10-30 (neighbors used for smoothing)
# Softmax temperature when computing neighbor weights for aggregation
DFS_ALPHA = 5.0  # recommended: 2.0-4.0
# Number of diffusion/smoothing steps (iterations)
DFS_STEPS = 3  # recommended: 1-5
# Interpolation parameter between original and aggregated embedding
DFS_BETA = 0.2  # recommended: 0.5-0.9 (higher -> keep original)
DFS_CHUNK_ROWS = 2048  # chunk size used for memory-efficient aggregation

@torch.no_grad()
@register("dfs")
def dfs_reranker(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    *,
    k: int = DFS_K,
    steps: int = DFS_STEPS,
    alpha: float = DFS_ALPHA,
    beta: float = DFS_BETA,
    chunk_rows: int = DFS_CHUNK_ROWS,
    metric: str = "cosine",
    prefer_gpu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diffusion-inspired Feature Smoothing (DFS) reranker.

    Args:
        q_feats: Query features tensor of shape (Nq, D).
        g_feats: Gallery features tensor of shape (Ng, D).
        k: Number of neighbors to use for aggregation.
        steps: Number of diffusion iterations.
        alpha: Softmax temperature for neighbor weighting.
        beta: Interpolation factor between original and aggregated embedding (0..1).
        chunk_rows: Chunk size for memory-efficient aggregation.
        metric: FAISS search metric, e.g. "cosine".
        prefer_gpu: Whether to use FAISS GPU index when available.

    Returns:
        q_emb: Smoothed query embeddings (Nq, D).
        g_emb: Smoothed gallery embeddings (Ng, D).

    Examples:
        >>> q_emb, g_emb = dfs_reranker(q_feats, g_feats, k=20, steps=3)
    """

    q = F.normalize(q_feats.float(), p=2, dim=1)
    g = F.normalize(g_feats.float(), p=2, dim=1)

    features = torch.cat((q, g), dim=0)
    total = features.shape[0]
    k_eff = max(1, min(k, total))
    diffusion_steps = max(1, steps)

    sim, idx = faiss_knn_topk(
        features,
        k=k_eff,
        metric=metric,
        prefer_gpu=prefer_gpu,
    )

    embeddings = features
    for _ in range(diffusion_steps):
        aggregated = aggregate_neighbors_chunked(
            embeddings,
            idx,
            sim,
            alpha=alpha,
            chunk_rows=chunk_rows,
        )
        embeddings = F.normalize(beta * embeddings + (1.0 - beta) * aggregated, p=2, dim=1)

    q_emb = embeddings[: q.shape[0]]
    g_emb = embeddings[q.shape[0] :]
    return q_emb, g_emb
