"""Cross-view UAV retrieval using the existing released benchmark manifests."""

import hashlib
import json
from pathlib import Path

import numpy as np

from gmlab.datasets.utils.retrieval_metadata import RetrievalImages


def sues200_retrieval(
    metadata_path,
    transform,
    *,
    split,
    direction="drone2satellite",
    height=None,
    image_root=None,
    gallery_limit=None,
    query_limit=None,
):
    """Read SUES-200 mixed heights, or one of 150/200/250/300 metres.

    Use the existing manifest's gallery, held-out queries and GT unchanged.
    Items supply images/index; dataset-level eval_meta supplies gallery-position GT.
    """
    if height is not None and (
        type(height) is not int or height not in {150, 200, 250, 300}
    ):
        raise ValueError("SUES-200 height must be null, 150, 200, 250 or 300")
    prefix = f"sues200_{direction}" + ("" if height is None else f"_{height}")
    return _verified_retrieval_images(
        metadata_path,
        prefix,
        transform,
        split=split,
        direction=direction,
        image_root=image_root,
        gallery_limit=gallery_limit,
        query_limit=query_limit,
    )


def denseuav_retrieval(
    metadata_path,
    transform,
    *,
    split,
    direction="drone2satellite",
    image_root=None,
    gallery_limit=None,
    query_limit=None,
):
    """Read DenseUAV's scene-ID protocol over all released heights and dates.

    Items supply images/index; dataset-level eval_meta supplies gallery-position GT.
    Both directions use the existing checksummed benchmark manifests.
    """
    return _verified_retrieval_images(
        metadata_path,
        f"denseuav_{direction}",
        transform,
        split=split,
        direction=direction,
        image_root=image_root,
        gallery_limit=gallery_limit,
        query_limit=query_limit,
    )


def _verified_retrieval_images(
    metadata_path, prefix, transform, *, split, direction, **options
):
    """Verify a benchmark manifest before decoding its common image/GT files.

    The JSON binds dataset name, counts, rectangular positive GT and per-file
    SHA256. Optional prefix limits affect the returned protocol only; verification
    always covers the full original annotation files. Images remain lazily read.
    """
    directions = {
        "drone2satellite": ("drone", "satellite"),
        "satellite2drone": ("satellite", "drone"),
    }
    if direction not in directions:
        raise ValueError(f"Unsupported cross-view retrieval direction: {direction}")
    root = Path(metadata_path)
    manifest_path = root / f"{prefix}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["dataset"] != prefix or manifest["direction"] != direction:
        raise ValueError("Retrieval manifest dataset/direction mismatch")
    arrays = []
    for suffix in ("dbImages", "qImages", "gt"):
        path = root / f"{prefix}_{suffix}.npy"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["files"][path.name]
        ):
            raise ValueError(f"Retrieval metadata checksum mismatch: {path}")
        arrays.append(np.load(path, allow_pickle=False))
    gallery, queries, gt = arrays
    if (len(gallery), len(queries)) != (
        manifest["num_references"],
        manifest["num_queries"],
    ):
        raise ValueError("Retrieval manifest counts mismatch")
    if gt.shape != (len(queries), manifest["positives_per_query"]):
        raise ValueError("Retrieval manifest positive count mismatch")
    query_modality, gallery_modality = directions[direction]
    dataset = RetrievalImages(
        root,
        prefix,
        transform,
        split=split,
        query_modality=query_modality,
        gallery_modality=gallery_modality,
        **options,
    )
    dataset.metadata.update(
        protocol_id=manifest["protocol_id"],
        protocol_source=manifest["protocol_source"],
        metadata_manifest=str(manifest_path),
        metadata_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        direction=direction,
    )
    return dataset
