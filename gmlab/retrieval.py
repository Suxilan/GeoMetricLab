"""Exact descriptor retrieval and pair construction, independent of CLI and storage."""

from typing import Sequence
from collections.abc import Mapping

import numpy as np
import torch
from .search import exact_search_chunks


def retrieve_chunks(queries, gallery, *, k, reranking=None, **search_options):
    """Yield (query offset, CPU int64 gallery positions [B,min(k,G)]).

    Inputs are global descriptors [Q,D]/[G,D] from a live model or a cache.
    Reranking is None, a registered name/callable, or {method: name, ...params}.
    Refinement runs once on the complete query/gallery collections before exact
    search is chunked. No GT, cache writes, identity inference or self exclusion
    occurs here. Transductive methods depend on the supplied query population.
    Search options are those of exact_search_chunks; k is clamped to G.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    queries = torch.as_tensor(queries).detach().float()
    gallery = torch.as_tensor(gallery).detach().float()
    if reranking is not None:
        from .rerank import apply_reranking

        options = (
            dict(reranking) if isinstance(reranking, Mapping) else {"method": reranking}
        )
        if "method" not in options:
            raise ValueError("reranking requires an explicit method")
        queries, gallery = apply_reranking(
            options.pop("method"), queries, gallery, **options
        )
    yield from exact_search_chunks(
        queries, gallery, k=min(k, len(gallery)), **search_options
    )


def retrieve(queries, gallery, *, k=10, reranking=None, **search_options):
    """Return CPU int64 gallery positions [Q,min(k,G)] in ranked order.

    The caller maps positions to gallery IDs/names. Use retrieve_chunks to bound
    result memory for large collections. Algorithm and input contracts are shared.
    """
    chunks = [
        rows
        for _, rows in retrieve_chunks(
            queries, gallery, k=k, reranking=reranking, **search_options
        )
    ]
    return (
        torch.cat(chunks)
        if chunks
        else torch.empty((0, min(k, len(gallery))), dtype=torch.long)
    )


def topk_neighbors(
    features: torch.Tensor, topk: int, prefer_gpu: bool = True, *, reranking=None
) -> np.ndarray:
    """Return cosine neighbors [N,min(topk+1,N)], including possible self rows."""
    if type(topk) is not int or topk < 1:
        raise ValueError("topk must be a positive integer")
    chunks = retrieve_chunks(
        features,
        features,
        k=min(topk + 1, len(features)),
        backend="auto",
        prefer_gpu=prefer_gpu,
        reranking=reranking,
    )
    return torch.cat([rows for _, rows in chunks]).numpy()


def build_colmap_pairs(
    relative_names: Sequence[str], neighbor_indices: np.ndarray, topk: int
) -> list[tuple[str, str]]:
    if type(topk) is not int or topk < 1:
        raise ValueError("topk must be a positive integer")
    neighbor_indices = np.asarray(neighbor_indices)
    if (
        neighbor_indices.ndim != 2
        or neighbor_indices.shape[0] != len(relative_names)
        or neighbor_indices.dtype.kind not in "iu"
        or (neighbor_indices < 0).any()
        or (neighbor_indices >= len(relative_names)).any()
    ):
        raise ValueError("Neighbor indices must be integer [N,K] in the name row space")
    seen_pairs: set[tuple[str, str]] = set()
    pair_list: list[tuple[str, str]] = []

    for query_index, neighbors in enumerate(neighbor_indices):
        kept = 0
        query_name = relative_names[query_index]
        for neighbor_index in neighbors:
            if neighbor_index == query_index:
                continue
            ref_name = relative_names[int(neighbor_index)]
            pair = tuple(sorted((query_name, ref_name)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pair_list.append(pair)
            kept += 1
            if kept >= topk:
                break

    return pair_list


def aggregate_vlad(values, centers, index):
    """Encode float32 [N,D] against centers [K,D] and a matching L2 index.

    Hard assignment, per-cluster residual L2 normalization, then global L2.
    Empty clusters remain zero. The caller owns the index and its resource policy.
    """
    values, centers = np.asarray(values), np.asarray(centers)
    if (
        values.ndim != 2
        or centers.ndim != 2
        or not centers.shape[0]
        or values.shape[1] != centers.shape[1]
        or not np.isfinite(values).all()
        or not np.isfinite(centers).all()
    ):
        raise ValueError("Expected finite descriptors [N,D] and centers [K,D]")
    if not len(values):
        raise ValueError("VLAD cannot encode an image without SIFT descriptors")
    labels = index.search(values, 1)[1][:, 0]
    if (
        labels.shape != (len(values),)
        or (labels < 0).any()
        or (labels >= len(centers)).any()
    ):
        raise ValueError("Assignment index returned invalid cluster labels")
    residuals = np.zeros_like(centers)
    np.add.at(residuals, labels, values - centers[labels])
    residuals /= np.maximum(np.linalg.norm(residuals, axis=1, keepdims=True), 1e-12)
    vector = residuals.ravel()
    return vector / max(np.linalg.norm(vector), 1e-12)


def sample_local_descriptors(images, *, per_image, seed):
    """Sample at most per_image rows from each [Ni,D] image in caller order.

    Return float32 [sum(min(Ni,per_image)),D]. This preserves the historical
    SIFT-VLAD per-image sampling distribution and RNG order. Memory scales with
    the selected descriptors, not all source descriptors. It is not a global cap.
    """
    if per_image < 1:
        raise ValueError("per_image must be positive")
    random = np.random.default_rng(seed)
    sampled, dimension = [], None
    for values in images:
        values = _descriptor_matrix(values, dimension)
        dimension = values.shape[1]
        take = min(per_image, len(values))
        sampled.append(values[random.choice(len(values), take, replace=False)])
    if not sampled or not any(len(part) for part in sampled):
        raise ValueError("No training descriptors available")
    return np.ascontiguousarray(np.concatenate(sampled), dtype=np.float32)


def _descriptor_matrix(values, dimension=None):
    values = np.asarray(values, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] < 1
        or (dimension is not None and values.shape[1] != dimension)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected finite descriptor chunks [N,D] with one dimension")
    return values


def spatial_descriptors(features):
    """Convert a finite [B,C,H,W] chunk to row-major [B*H*W,C] patches.

    The caller obtains C/H/W from the public reader. No token-grid inference,
    normalization, sampling or IO is performed; rectangular grids are retained.
    """
    features = np.asarray(features)
    if features.ndim != 4 or any(d < 1 for d in features.shape):
        raise ValueError("Expected a nonempty spatial chunk [B,C,H,W]")
    if not np.isfinite(features).all():
        raise ValueError("Spatial descriptors must be finite")
    return np.ascontiguousarray(
        features.transpose(0, 2, 3, 1).reshape(-1, features.shape[1]), dtype=np.float32
    )


def sample_descriptor_chunks(chunks, *, max_descriptors, seed, normalize=False):
    """Uniformly sample a bounded float32 [M,D] from ordered descriptor chunks.

    Assign independent random priorities to rows and retain the smallest M.
    Selection is invariant to chunk boundaries for the same ordered rows/seed.
    Return rows in source order; memory is bounded by M plus one input chunk.
    Source ordering and sampling budget are explicit consumer choices.
    """
    if max_descriptors < 1:
        raise ValueError("max_descriptors must be positive")
    random = np.random.default_rng(seed)
    selected = priorities = positions = None
    offset = 0
    for values in chunks:
        values = _descriptor_matrix(
            values, None if selected is None else selected.shape[1]
        )
        if not len(values):
            continue
        scores = random.random(len(values))
        rows = np.arange(offset, offset + len(values))
        offset += len(values)
        if selected is not None:
            values = np.concatenate((selected, values))
            scores = np.concatenate((priorities, scores))
            rows = np.concatenate((positions, rows))
        if len(values) > max_descriptors:
            keep = np.argpartition(scores, max_descriptors - 1)[:max_descriptors]
            values, scores, rows = values[keep], scores[keep], rows[keep]
        selected, priorities, positions = values, scores, rows
    if selected is None:
        raise ValueError("No training descriptors available")
    selected = np.ascontiguousarray(selected[np.argsort(positions)], dtype=np.float32)
    if normalize:
        selected /= np.maximum(np.linalg.norm(selected, axis=1, keepdims=True), 1e-12)
    return selected


def fit_kmeans(
    descriptors,
    *,
    num_clusters,
    iterations,
    seed,
    min_points_per_centroid=39,
    max_points_per_centroid=None,
):
    """Explicitly fit CPU FAISS KMeans and return finite centers [K,D].

    This numerical function never discovers data, loads files or triggers feature
    extraction. Callers select/sample inputs and set FAISS thread policy explicitly.
    Defaults retain the existing SIFT-VLAD requirement of 39 descriptors per center.
    max_points_per_centroid=None retains FAISS's default; spatial consumers can
    explicitly pass their historical cap without changing the SIFT recipe.
    """
    values = _descriptor_matrix(descriptors)
    if num_clusters < 1 or iterations < 1 or min_points_per_centroid < 1:
        raise ValueError(
            "KMeans cluster, iteration and minimum-point counts must be positive"
        )
    minimum = num_clusters * min_points_per_centroid
    if len(values) < minimum:
        raise ValueError(
            f"Need at least {minimum} training descriptors for {num_clusters} centers"
        )
    import faiss

    options = {"niter": iterations, "nredo": 1, "seed": seed, "gpu": False}
    if min_points_per_centroid != 39:
        options["min_points_per_centroid"] = min_points_per_centroid
    if max_points_per_centroid is not None:
        if max_points_per_centroid < 1:
            raise ValueError("max_points_per_centroid must be positive")
        options["max_points_per_centroid"] = max_points_per_centroid
    kmeans = faiss.Kmeans(values.shape[1], num_clusters, **options)
    kmeans.train(np.ascontiguousarray(values))
    centers = _descriptor_matrix(kmeans.centroids, values.shape[1])
    if centers.shape[0] != num_clusters:
        raise ValueError("KMeans returned an unexpected center count")
    return np.ascontiguousarray(centers)
