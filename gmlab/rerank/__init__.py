"""Global descriptor refinement before final retrieval ranking."""

from .rerankers import apply_reranking, register

__all__ = ["apply_reranking", "register"]
