"""Protocol-explicit retrieval with shared exact search and independent GT semantics."""

import torch
from ..retrieval import retrieve_chunks


def _role_indices(values, num_samples, device):
    indices = torch.as_tensor(values, device=device)
    if (
        indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
        or not len(indices)
    ):
        raise ValueError("Query/gallery indices must be nonempty integer vectors")
    if (
        indices.min() < 0
        or indices.max() >= num_samples
        or len(indices.unique()) != len(indices)
    ):
        raise ValueError("Invalid or repeated manifest indices within a role")
    return indices


def _positive_sets(ground_truth, query_indices, gallery_indices, exclude_self):
    if ground_truth is None or len(ground_truth) != len(query_indices):
        raise ValueError("Evaluation requires one GT row per query")
    positives = []
    for query, row in enumerate(ground_truth):
        values = torch.as_tensor(row)
        if values.ndim != 1 or (
            values.numel() and values.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("GT rows must contain integer gallery positions")
        positive = set(values.tolist())
        if any(value < 0 or value >= len(gallery_indices) for value in positive):
            raise ValueError("GT gallery position out of range")
        if exclude_self and any(
            int(gallery_indices[value]) == int(query_indices[query])
            for value in positive
        ):
            raise ValueError("Self image must not appear in positive GT")
        positives.append(positive)
    return positives


def _search_options(k_values, metric, tie_break, query_chunk_size):
    ks = sorted(set(k_values))
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("k_values must contain positive integers")
    if metric not in {"cosine", "l2"} or tie_break not in {"native", "torch", "index"}:
        raise ValueError("Unknown distance or tie policy")
    if type(query_chunk_size) is not int or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    return ks


def _accumulate_query(totals, ranking, positive, ks, full_ranking):
    """Accumulate macro numerators; AP@k retains all positives as denominator."""
    hits = [(rank, value) for rank, value in enumerate(ranking, 1) if value in positive]
    precisions = [(rank, count / rank) for count, (rank, _) in enumerate(hits, 1)]
    if full_ranking:
        totals["mAP"] += sum(precision for _, precision in precisions) / len(positive)
    for k in ks:
        totals[f"mAP@{k}"] += sum(
            precision for rank, precision in precisions if rank <= k
        ) / len(positive)
        totals[f"R@{k}"] += sum(rank <= k for rank, _ in precisions) / len(positive)
        totals[f"CMC@{k}"] += float(bool(hits) and hits[0][0] <= k)


def evaluate_retrieval(
    embeddings,
    *,
    query_indices,
    gallery_indices,
    ground_truth,
    exclude_self=False,
    k_values=(1, 5, 10),
    metric="cosine",
    full_ranking=True,
    tie_break="native",
    query_chunk_size=128,
    search_backend="auto",
    faiss_threshold=4000,
    prefer_gpu=True,
    gpu_device=None,
    reranking=None,
):
    """Compute macro percentages using the project's existing metric definitions.

    embeddings [N,D] are in stable manifest order. query/gallery_indices address
    that manifest; each GT row lists *gallery positions*, not manifest indices.
    R@k is positive coverage, CMC@k is any-positive hit rate. AP@k divides by all
    query positives; mAP requires full ranking. Queries without positives are
    skipped, with zero metrics when none are valid. Self GT is rejected when
    excluding self, instead of silently changing its denominator.

    Default native ties follow the automatically selected exact-search backend.
    tie_break='torch' preserves the old torch.topk behavior on the same device
    and search width (ties have no portable order); 'index' explicitly chooses
    ascending gallery position for equal scores. k beyond gallery size is
    clamped for search, while output names retain the requested k.
    """
    if type(exclude_self) is not bool or type(full_ranking) is not bool:
        raise TypeError("exclude_self and full_ranking must be bool")
    x = torch.as_tensor(embeddings).detach().float()
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("Expected finite embeddings [N,D]")
    qids = _role_indices(query_indices, len(x), x.device)
    gids = _role_indices(gallery_indices, len(x), x.device)
    positives = _positive_sets(ground_truth, qids, gids, exclude_self)
    ks = _search_options(k_values, metric, tie_break, query_chunk_size)
    gallery = x[gids]
    queries = x[qids]
    search_k = (
        len(gids) if full_ranking else min(len(gids), max(ks) + int(exclude_self))
    )
    names = [f"{name}@{k}" for k in ks for name in ("mAP", "R", "CMC")]
    totals = dict.fromkeys(names + (["mAP"] if full_ranking else []), 0.0)
    valid = sum(bool(positive) for positive in positives)
    # Ranking entries are gallery positions. Resolve physical self identities
    # once instead of converting two Tensor scalars per candidate in Python.
    gallery_positions = {
        identity: index for index, identity in enumerate(gids.tolist())
    }
    self_positions = [gallery_positions.get(identity) for identity in qids.tolist()]
    for start, rankings in retrieve_chunks(
        queries,
        gallery,
        metric=metric,
        tie_break=tie_break,
        k=search_k,
        backend=search_backend,
        faiss_threshold=faiss_threshold,
        prefer_gpu=prefer_gpu,
        gpu_device=gpu_device,
        chunk_size=query_chunk_size,
        reranking=reranking,
    ):
        for offset, ranking in enumerate(rankings.tolist()):
            query = start + offset
            if not positives[query]:
                continue
            if exclude_self:
                self_position = self_positions[query]
                ranking = [value for value in ranking if value != self_position]
            _accumulate_query(totals, ranking, positives[query], ks, full_ranking)
    return {
        **{key: value * 100 / max(valid, 1) for key, value in totals.items()},
        "num_valid_queries": float(valid),
        "num_queries_total": float(len(qids)),
    }
