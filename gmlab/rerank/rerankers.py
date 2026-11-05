"""Explicit, lazy reranking dispatch; algorithms retain configurable parameters."""

from collections.abc import Callable
from importlib import import_module

import torch

_BUILTINS = {name: f"{name}_reranker" for name in ("aqe", "dba", "dfs", "gnn", "krnn")}
RERANK_REGISTRY: dict[str, Callable] = {}


def _validate_descriptors(queries, gallery):
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or not value.is_floating_point()
        or not len(value)
        or value.shape[1] == 0
        or not torch.isfinite(value).all()
        for value in (queries, gallery)
    ):
        raise ValueError("Expected nonempty finite floating [Q,D]/[G,D] tensors")
    if queries.shape[1] != gallery.shape[1] or queries.device != gallery.device:
        raise ValueError("Query/gallery descriptors must share width and device")


def register(name: str):
    """Register an external descriptor refiner without replacing existing names."""

    def decorate(function):
        key = name.lower()
        if key in RERANK_REGISTRY:
            raise ValueError(f"Reranker already registered: {key}")
        RERANK_REGISTRY[key] = function
        return function

    return decorate


@torch.no_grad()
def apply_reranking(method, query_feats, gallery_feats, **parameters):
    """Refine [Q,D]/[G,D] descriptors, preserving query/gallery row order.

    method is a registered algorithm name or an injected callable. Parameters
    are passed unchanged to that algorithm. Inputs must share feature width and
    device; output width may change (e.g. graph embeddings). No GT is consumed.
    """
    _validate_descriptors(query_feats, gallery_feats)
    if method is None or method == "none":
        if parameters:
            raise ValueError("Disabled reranking does not accept algorithm parameters")
        return query_feats, gallery_feats
    for name in ("k", "k1", "k2", "steps", "chunk_rows"):
        if name in parameters and (
            type(parameters[name]) is not int or parameters[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if callable(method):
        algorithm = method
    else:
        if not isinstance(method, str):
            raise TypeError("Reranking requires an algorithm name or callable")
        key = method.lower()
        if key not in RERANK_REGISTRY and key in _BUILTINS:
            import_module(f"gmlab.rerank.{_BUILTINS[key]}")
        if key not in RERANK_REGISTRY:
            raise ValueError(f"Unknown reranker: {method}")
        algorithm = RERANK_REGISTRY[key]
    result = algorithm(query_feats, gallery_feats, **parameters)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError(
            "Reranker must return (query_descriptors, gallery_descriptors)"
        )
    queries, gallery = result
    _validate_descriptors(queries, gallery)
    if len(queries) != len(query_feats) or len(gallery) != len(gallery_feats):
        raise ValueError("Reranking must preserve query/gallery row counts")
    if queries.device != query_feats.device:
        raise ValueError("Reranking must preserve the input device")
    return queries, gallery
