"""Reranking registry and dispatcher."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch

# Registry ---------------------------------------------------------------
RERANK_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable):
        RERANK_REGISTRY[name.lower()] = fn
        return fn

    return deco


def apply_reranking(method: str, query_feats: torch.Tensor, gallery_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch reranker by name. Returns (query_feats_new, gallery_feats_new)."""

    if method is None or method.lower() == "none":
        return query_feats, gallery_feats

    key = method.lower()
    if key not in RERANK_REGISTRY:
        raise ValueError(f"Unknown reranking method: {method}. Available: {list(RERANK_REGISTRY.keys())}")

    return RERANK_REGISTRY[key](query_feats, gallery_feats)


# Import built-in rerankers (registration happens via decorators)
try:
    from . import gnn_reranker  # noqa: F401
    from . import aqe_reranker  # noqa: F401
    from . import dba_reranker  # noqa: F401
    from . import krnn_reranker  # noqa: F401
    from . import dfs_reranker  # noqa: F401
except Exception:
    # Allow registry to exist even if optional rerankers fail to import
    pass
