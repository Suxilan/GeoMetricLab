"""U1652 cross-view retrieval from existing metadata; never reads train split."""

from collections.abc import Callable, Mapping
from pathlib import Path

from gmlab.datasets.utils.retrieval_metadata import RetrievalImages


def u1652_retrieval(
    metadata_path: str | Path,
    transform: Callable | Mapping | None,
    *,
    split: str,
    direction="drone2satellite",
    image_root: str | Path | None = None,
    gallery_limit: int | None = None,
    query_limit: int | None = None,
) -> RetrievalImages:
    """Build the requested view direction; item has images/index, GT is eval_meta."""
    directions = {
        "drone2satellite": ("drone", "satellite"),
        "satellite2drone": ("satellite", "drone"),
        "satellite2street": ("satellite", "street"),
        "street2satellite": ("street", "satellite"),
    }
    if direction not in directions:
        raise ValueError(f"Unsupported U1652 retrieval direction: {direction}")
    query_modality, gallery_modality = directions[direction]
    return RetrievalImages(
        metadata_path,
        f"u1652_{direction}",
        transform,
        same_collection=False,
        split=split,
        image_root=image_root,
        gallery_limit=gallery_limit,
        query_limit=query_limit,
        query_modality=query_modality,
        gallery_modality=gallery_modality,
    )
