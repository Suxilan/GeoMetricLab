"""GNN reranking implementation (kept separate from registry).

Uses chunked PyTorch propagation on the input device. The adjacency matrix
requires quadratic memory on the input device and host; no external CUDA
extension is loaded.
"""

# Adapted from Person_reID_baseline_pytorch/GPU-Re-Ranking (MIT).
# Copyright (c) 2018 Zhedong Zheng
# https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/GPU-Re-Ranking
# See LICENSES/GNN-ReRanking.txt and NOTICE for the retained license.

from __future__ import annotations

import torch
import torch.nn.functional as F

from .neighbors import knn_topk
from .rerankers import register

# Fixed hyperparameters (update here if needed)
# k1: number of nearest neighbors retrieved by FAISS (affects recall/candidate set)
# k2: number of neighbors used during propagation/aggregation (affects smoothing)
GNN_K1 = 50  # recommended: 20-60
GNN_K2 = 20  # recommended: 1-10


@torch.no_grad()
def _gnn_embeddings_torch(
    X_q: torch.Tensor, X_g: torch.Tensor, k1: int, k2: int
) -> tuple[torch.Tensor, torch.Tensor]:

    device = X_q.device
    query_num = X_q.shape[0]

    X_u = torch.cat((X_q, X_g), dim=0).contiguous()
    N = X_u.shape[0]

    k1_eff = min(k1, N)
    k2_eff = min(k2, k1_eff)
    S_cpu, I_cpu = knn_topk(X_u, k1_eff, metric="cosine", prefer_gpu=False)
    S = S_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
    I = I_cpu.to(device=device, dtype=torch.long, non_blocking=True)

    A = torch.zeros((N, N), device=device, dtype=torch.float32)
    A.scatter_(1, I, 1.0)

    S = S * S
    if k2_eff > 1:
        nbr = I[:, :k2_eff]
        w = S[:, :k2_eff]
        for _ in range(2):
            A = A + A.t()
            A_new = torch.zeros_like(A)
            for j in range(k2_eff):
                rows = nbr[:, j]
                Aj = A.index_select(0, rows)
                A_new.add_(Aj * w[:, j].unsqueeze(1))

            A = F.normalize(A_new, p=2, dim=1)

    return A[:query_num], A[query_num:]


@torch.no_grad()
def _gnn_embeddings_torch_chunked(
    X_q: torch.Tensor,
    X_g: torch.Tensor,
    k1: int,
    k2: int,
    chunk_rows: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked PyTorch graph propagation with quadratic host adjacency storage.

    Args:
        chunk_rows: Bound the propagation workspace on the input device.
    """
    device = X_q.device
    query_num = X_q.shape[0]

    X_u = torch.cat((X_q, X_g), dim=0).contiguous()
    N = X_u.shape[0]

    k1_eff = min(k1, N)
    k2_eff = min(k2, k1_eff)

    S, I = knn_topk(X_u, k1_eff, metric="cosine", prefer_gpu=True)
    del X_u

    S = S.to(device=device, dtype=torch.float32, non_blocking=True).square_()
    I = I.to(device=device, dtype=torch.long, non_blocking=True)

    A = torch.zeros((N, N), device=device, dtype=torch.float32)
    A.scatter_(1, I, 1.0)

    if k2_eff <= 1:
        return A[:query_num].clone(), A[query_num:].clone()

    nbr = I[:, :k2_eff].contiguous()
    wts = S[:, :k2_eff].contiguous()
    del I, S

    A_cpu = torch.empty(
        (N, N), device="cpu", dtype=torch.float32, pin_memory=X_q.is_cuda
    )

    for step in range(2):
        A_T = A.transpose(0, 1)

        for r0 in range(0, N, chunk_rows):
            r1 = min(N, r0 + chunk_rows)
            b = r1 - r0

            nbr_blk = nbr[r0:r1]  # [b,k2] on GPU
            w_blk = wts[r0:r1]  # [b,k2] on GPU
            flat = nbr_blk.reshape(-1)  # [b*k2]

            rows = A.index_select(0, flat)  # [b*k2, N]
            cols = A_T.index_select(0, flat)  # [b*k2, N]
            rows.add_(cols)
            sym = rows.view(b, k2_eff, N)  # [b,k2,N]

            out = torch.bmm(w_blk.unsqueeze(1), sym).squeeze(1)  # [b,N]
            out = F.normalize(out, p=2, dim=1)

            A_cpu[r0:r1].copy_(out, non_blocking=True)

        if X_q.is_cuda:
            torch.cuda.synchronize(X_q.device)
        A.copy_(A_cpu, non_blocking=True)

    res_q = A[:query_num].clone()
    res_g = A[query_num:].clone()

    del A, nbr, wts, A_cpu

    return res_q, res_g


@register("gnn")
def gnn_reranker(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    *,
    k1: int = GNN_K1,
    k2: int = GNN_K2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GNN-based reranker adapter.

    Normalize input features and run chunked propagation on their device.
    No native extension or implicit CUDA transfer is required.

    Args:
        q_feats: Query feature tensor of shape (Nq, D).
        g_feats: Gallery feature tensor of shape (Ng, D).
        k1: Number of nearest neighbors used by the FAISS stage.
        k2: Number of neighbors used for propagation/aggregation.

    Returns:
        q_emb: Graph embeddings (Nq, Nq + Ng), on the input device.
        g_emb: Graph embeddings (Ng, Nq + Ng), on the input device.

    Examples:
        >>> q_emb, g_emb = gnn_reranker(q_feats, g_feats, k1=56, k2=3)
    """

    # Normalize inputs first
    q = F.normalize(q_feats.float(), p=2, dim=1)
    g = F.normalize(g_feats.float(), p=2, dim=1)

    return _gnn_embeddings_torch_chunked(q, g, k1, k2, chunk_rows=4096)
