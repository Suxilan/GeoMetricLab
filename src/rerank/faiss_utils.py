"""Helpers for FAISS kNN lookups shared across rerankers."""

from __future__ import annotations

import numpy as np
import torch
import faiss


def faiss_knn_topk(
    query: torch.Tensor,
    k: int,
    metric: str = "cosine",
    *,
    database: torch.Tensor | None = None,
    prefer_gpu: bool = True,
    gpu_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Search for the top-k neighbors using FAISS (GPU when available)."""

    query_cpu = query.detach().cpu().float()
    if database is None:
        database_cpu = query_cpu
    else:
        database_cpu = database.detach().cpu().float()

    query_np = np.ascontiguousarray(query_cpu.numpy(), dtype=np.float32)
    data_np = np.ascontiguousarray(database_cpu.numpy(), dtype=np.float32)
    d = data_np.shape[1]

    if metric == "cosine":
        faiss.normalize_L2(data_np)
        faiss.normalize_L2(query_np)
        index_cpu = faiss.IndexFlatIP(d)
    else:
        index_cpu = faiss.IndexFlatL2(d)

    index = index_cpu
    if prefer_gpu and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0:
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, gpu_id, index_cpu)

    index.add(data_np)
    distances, indices = index.search(query_np, k)
    return torch.from_numpy(distances), torch.from_numpy(indices)

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
    base_feats: [Ng,D]（建议已 normalize）
    neighbor_indices: [Nq,k]（CPU/GPU都行）
    similarities: [Nq,k]（CPU/GPU都行）
    返回: [Nq,D] float32
    """
    device = base_feats.device
    Ng, D = base_feats.shape
    Nq, k = neighbor_indices.shape

    idx = neighbor_indices.to(device=device, dtype=torch.long, non_blocking=True)
    sim = similarities.to(device=device, dtype=torch.float32, non_blocking=True)

    out = torch.empty((Nq, D), device=device, dtype=torch.float32)

    for r0 in range(0, Nq, chunk_rows):
        r1 = min(Nq, r0 + chunk_rows)
        idx_blk = idx[r0:r1]     # [b,k]
        sim_blk = sim[r0:r1]     # [b,k]

        w = torch.softmax(sim_blk * alpha, dim=1)  # [b,k] float32
        neigh = base_feats.index_select(0, idx_blk.reshape(-1)).reshape(r1 - r0, k, D)
        qe = torch.bmm(w.unsqueeze(1).to(neigh.dtype), neigh).squeeze(1)
        out[r0:r1] = qe.float()

    return out