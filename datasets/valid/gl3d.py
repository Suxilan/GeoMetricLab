"""GL3D retrieval with explicit same-image exclusion and query role mapping."""

from collections.abc import Callable, Mapping
from pathlib import Path

from gmlab.datasets.utils.retrieval_metadata import RetrievalImages


def gl3d_retrieval(
    metadata_path: str | Path,
    transform: Callable | Mapping | None,
    *,
    split: str,
    prefix="gl3d_test",
    image_root: str | Path | None = None,
    gallery_limit: int | None = None,
    query_limit: int | None = None,
) -> RetrievalImages:
    """Read image/role/GT manifests; no train sampler is constructed."""
    return RetrievalImages(
        metadata_path,
        prefix,
        transform,
        same_collection=True,
        split=split,
        image_root=image_root,
        gallery_limit=gallery_limit,
        query_limit=query_limit,
    )
