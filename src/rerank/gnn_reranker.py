"""GNN reranking implementation (kept separate from registry).

Provides a CUDA extension path when available, and falls back to a pure PyTorch
implementation to avoid CUDA illegal memory access or OOM on large batches.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from .rerankers import register
from .faiss_utils import faiss_knn_topk

# Fixed hyperparameters (update here if needed)
# k1: number of nearest neighbors retrieved by FAISS (affects recall/candidate set)
# k2: number of neighbors used during propagation/aggregation (affects smoothing)
GNN_K1 = 50  # recommended: 20-60
GNN_K2 = 20   # recommended: 1-10

# Dense CUDA extension path is deprecated for large-scale runs because it still
# constructs dense N×N matrices. Keep it off by default.

_RERANK_ROOT = Path(__file__).resolve().parents[2] / "GPU-Re-Ranking"
if _RERANK_ROOT.is_dir() and str(_RERANK_ROOT) not in sys.path:
    sys.path.append(str(_RERANK_ROOT))

try:
    build_adjacency_matrix = importlib.import_module("build_adjacency_matrix")
    gnn_propagate = importlib.import_module("gnn_propagate")
except Exception:
    build_adjacency_matrix = None
    gnn_propagate = None

def _gnn_embeddings_cuda(X_q: torch.Tensor, X_g: torch.Tensor, k1: int, k2: int) -> Tuple[torch.Tensor, torch.Tensor]:
    query_num = X_q.shape[0]
    X_u = torch.cat((X_q, X_g), dim=0)
    N = X_u.shape[0]
    k1_eff = min(k1, N)
    k2_eff = min(k2, k1_eff)

    S_cpu, initial_rank_cpu = faiss_knn_topk(X_u, k1_eff, metric="cosine", prefer_gpu=True)
    S = S_cpu.to(device=X_u.device, dtype=torch.float32, non_blocking=True)
    initial_rank = initial_rank_cpu.to(device=X_u.device, dtype=torch.long, non_blocking=True)

    A = build_adjacency_matrix.forward(initial_rank.contiguous().float())
    S = S * S

    if k2_eff > 1:
        for _ in range(2):
            A = A + A.T
            A = gnn_propagate.forward(
                A.contiguous(),
                initial_rank[:, :k2_eff].contiguous().float(),
                S[:, :k2_eff].contiguous().float(),
            )
            A = F.normalize(A, p=2, dim=1)

    return A[:query_num], A[query_num:]

@torch.no_grad()
def _gnn_embeddings_torch(
    X_q: torch.Tensor,
    X_g: torch.Tensor,
    k1: int,
    k2: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    device = X_q.device
    query_num = X_q.shape[0]

    X_u = torch.cat((X_q, X_g), dim=0).contiguous()
    N = X_u.shape[0]

    k1_eff = min(k1, N)
    k2_eff = min(k2, k1_eff)
    S_cpu, I_cpu = faiss_knn_topk(X_u, k1_eff, metric="cosine", prefer_gpu=False)
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GNN Re-ranking 的纯 PyTorch 分块优化实现。
    
    Args:
        chunk_rows: 分块大小，显存越小该值应设置越小 (e.g., 256 or 512)。
    """
    device = X_q.device
    query_num = X_q.shape[0]

    X_u = torch.cat((X_q, X_g), dim=0).contiguous()
    N = X_u.shape[0]

    k1_eff = min(k1, N)
    k2_eff = min(k2, k1_eff)

    S, I = faiss_knn_topk(X_u, k1_eff, metric="cosine", prefer_gpu=True)
    del X_u

    S = S.to(device=device, dtype=torch.float32, non_blocking=True).square_()
    I = I.to(device=device, dtype=torch.long, non_blocking=True)

    A = torch.zeros((N, N), device=device, dtype=torch.float32)
    A.scatter_(1, I, 1.0)

    if k2_eff <= 1:
        return A[:query_num].cpu(), A[query_num:].cpu()

    nbr = I[:, :k2_eff].contiguous()
    wts = S[:, :k2_eff].contiguous()
    del I, S

    A_cpu = torch.empty((N, N), device="cpu", dtype=torch.float32, pin_memory=True)

    for step in range(2):
        print(f"GNN propagation step {step + 1}/2 (GPU Compute)...")
        A_T = A.transpose(0, 1)

        for r0 in range(0, N, chunk_rows):
            r1 = min(N, r0 + chunk_rows)
            b = r1 - r0

            nbr_blk = nbr[r0:r1]                 # [b,k2] on GPU
            w_blk = wts[r0:r1]                   # [b,k2] on GPU
            flat = nbr_blk.reshape(-1)           # [b*k2]

            rows = A.index_select(0, flat)       # [b*k2, N]
            cols = A_T.index_select(0, flat)     # [b*k2, N]
            rows.add_(cols)
            sym = rows.view(b, k2_eff, N)        # [b,k2,N]

            out = torch.bmm(w_blk.unsqueeze(1), sym).squeeze(1)  # [b,N]
            out = F.normalize(out, p=2, dim=1)

            A_cpu[r0:r1].copy_(out, non_blocking=True)

        torch.cuda.synchronize()
        A.copy_(A_cpu, non_blocking=True)

    print("GNN finished. Moving results to CPU to avoid OOM in metrics...", flush=True)
    res_q = A[:query_num].cpu()
    res_g = A[query_num:].cpu()

    del A, nbr, wts, A_cpu
    torch.cuda.empty_cache()

    return res_q, res_g

@register("gnn")
def gnn_reranker(
    q_feats: torch.Tensor,
    g_feats: torch.Tensor,
    *,
    k1: int = GNN_K1,
    k2: int = GNN_K2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GNN-based reranker adapter.

    This function normalizes input features, runs the chunked GPU/CPU-aware
    propagation implementation and returns refined embeddings for queries and
    gallery. It preserves the existing fallback behaviour: if the optimized
    chunked path fails, it falls back to the CPU-only propagation routine.

    Args:
        q_feats: Query feature tensor of shape (Nq, D).
        g_feats: Gallery feature tensor of shape (Ng, D).
        k1: Number of nearest neighbors used by the FAISS stage.
        k2: Number of neighbors used for propagation/aggregation.

    Returns:
        q_emb: Refined query embeddings (Nq, D).
        g_emb: Refined gallery embeddings (Ng, D).

    Examples:
        >>> q_emb, g_emb = gnn_reranker(q_feats, g_feats, k1=56, k2=3)
    """

    # Normalize inputs first
    q = F.normalize(q_feats.float(), p=2, dim=1)
    g = F.normalize(g_feats.float(), p=2, dim=1)

    device = q.device if q.is_cuda else torch.device("cuda")
    q_cuda = q.to(device, non_blocking=True)
    g_cuda = g.to(device, non_blocking=True)
    try:
        q_emb, g_emb = _gnn_embeddings_torch_chunked(q_cuda, g_cuda, k1, k2, chunk_rows=4096)
    except Exception:
        # Keep fallback behaviour but do not alter computation logic
        torch.cuda.empty_cache()
        q_cpu = q.cpu()
        g_cpu = g.cpu()
        q_emb, g_emb = _gnn_embeddings_torch(q_cpu, g_cpu, k1, k2)

    return q_emb, g_emb
