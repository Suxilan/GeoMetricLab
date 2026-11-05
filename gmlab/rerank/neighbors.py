"""Exact neighbors and weighted aggregation for global descriptor refinement."""

from __future__ import annotations

import torch


def knn_topk(
    query: torch.Tensor,
    k: int,
    metric: str = "cosine",
    *,
    database: torch.Tensor | None = None,
    prefer_gpu: bool = True,
    gpu_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU affinities and indices [Q,k] using shared exact search.

    Affinities are cosine similarity or negative squared L2 distance, so larger
    always means closer. Backend selection follows gmlab.search; FAISS is optional.
    """

    from torch.nn import functional as F

    from ..search import exact_search_chunks

    q = query.detach().float()
    g = q if database is None else database.detach().to(q.device).float()
    indices = torch.cat(
        [
            rows
            for _, rows in exact_search_chunks(
                q,
                g,
                k=k,
                metric=metric,
                prefer_gpu=prefer_gpu,
                gpu_device=gpu_id,
            )
        ]
    )
    q, g = q.cpu(), g.cpu()
    if metric == "cosine":
        q, g = F.normalize(q, dim=1), F.normalize(g, dim=1)
    scores = []
    for start in range(0, len(q), 128):
        selected = g[indices[start : start + 128]]
        current = q[start : start + 128, None]
        score = (
            (current * selected).sum(-1)
            if metric == "cosine"
            else -(current - selected).square().sum(-1)
        )
        scores.append(score)
    return torch.cat(scores), indices


@torch.no_grad()
def aggregate_neighbors(
    base_feats: torch.Tensor,
    neighbor_indices: torch.Tensor,
    similarities: torch.Tensor,
    alpha: float = 3.0,
) -> torch.Tensor:
    """Weighted aggregation of neighbors for query/gallery augmentation."""

    device = base_feats.device
    idx = neighbor_indices.to(device=device)
    sim = similarities.to(device=device)
    weights = torch.softmax(sim * alpha, dim=1)
    gathered = base_feats[idx]
    return torch.sum(weights.unsqueeze(-1) * gathered, dim=1)


@torch.no_grad()
def aggregate_neighbors_chunked(
    base_feats: torch.Tensor,
    neighbor_indices: torch.Tensor,
    similarities: torch.Tensor,
    *,
    alpha: float = 3.0,
    chunk_rows: int = 4096,
) -> torch.Tensor:
    """
    base_feats is [Ng,D]; indices and similarities are [Nq,k].
    Return float32 [Nq,D] on the input feature device.
    """
    device = base_feats.device
    Ng, D = base_feats.shape
    Nq, k = neighbor_indices.shape

    idx = neighbor_indices.to(device=device, dtype=torch.long, non_blocking=True)
    sim = similarities.to(device=device, dtype=torch.float32, non_blocking=True)

    out = torch.empty((Nq, D), device=device, dtype=torch.float32)

    for r0 in range(0, Nq, chunk_rows):
        r1 = min(Nq, r0 + chunk_rows)
        idx_blk = idx[r0:r1]  # [b,k]
        sim_blk = sim[r0:r1]  # [b,k]

        w = torch.softmax(sim_blk * alpha, dim=1)  # [b,k] float32
        neigh = base_feats.index_select(0, idx_blk.reshape(-1)).reshape(r1 - r0, k, D)
        qe = torch.bmm(w.unsqueeze(1).to(neigh.dtype), neigh).squeeze(1)
        out[r0:r1] = qe.float()

    return out
