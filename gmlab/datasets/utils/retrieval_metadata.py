"""Shared image manifests and static retrieval annotation decoding."""

import hashlib
from pathlib import Path

import numpy as np

from .image_list import ImageList
from .identities import integer_tensor


def _image_names(path: Path) -> list[str]:
    """Read a one-dimensional image-name manifest without coercing its values."""
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1:
        raise ValueError(f"Image manifest must be one-dimensional: {path}")
    names = values.tolist()
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"Image manifest must contain nonempty strings: {path}")
    return names


class RetrievalImages(ImageList):
    """Images + stable index, with retrieval roles/GT separate in eval_meta.

    Annotation files use the existing *_dbImages/qImages/gt.npy convention.
    image_root may differ from metadata_path. GT rows address gallery positions.
    Explicit prefix limits define a smaller candidate protocol and filter GT to
    that gallery; this is recorded as a subset, never reported as full retrieval.
    annotations_sha256 identifies annotation files plus subset/self options;
    it excludes model, preprocessing, distance, ranking and runtime versions.
    A same_collection manifest shares stable indices for identical image names.
    This shared utility does not import concrete train/valid readers.
    """

    def __init__(
        self,
        metadata_path,
        prefix,
        transform,
        *,
        image_root=None,
        same_collection=False,
        gallery_limit=None,
        query_limit=None,
        split,
        gallery_modality=None,
        query_modality=None,
    ):
        root = Path(metadata_path)
        files = [
            root / f"{prefix}_{suffix}.npy" for suffix in ("dbImages", "qImages", "gt")
        ]
        gallery = _image_names(files[0])
        queries = _image_names(files[1])
        gt = np.load(files[2], allow_pickle=True).tolist()
        if not isinstance(gt, list):
            raise ValueError("GT must contain a sequence of query annotation rows")
        if len(gt) != len(queries):
            raise ValueError("One annotation row is required per query")
        for limit in (gallery_limit, query_limit):
            if limit is not None:
                value = integer_tensor(limit, name="Prefix limits", minimum=1)
                if value.ndim != 0:
                    raise ValueError("Prefix limits must be scalar positive integers")
        original_gallery_size = len(gallery)
        gallery = gallery[:gallery_limit]
        queries = queries[:query_limit]
        validated_gt = []
        for row in gt:
            positions = integer_tensor(
                row,
                name="Source GT gallery positions",
                minimum=0,
                maximum=original_gallery_size - 1,
            )
            if positions.ndim != 1:
                raise ValueError("Source GT gallery positions must be a vector")
            validated_gt.append(positions.tolist())
        gt = [
            tuple(v for v in row if v < len(gallery))
            for row in validated_gt[: len(queries)]
        ]
        if same_collection:
            if len(set(gallery)) != len(gallery):
                raise ValueError(
                    "Same-collection gallery names must identify unique images"
                )
            positions = {name: i for i, name in enumerate(gallery)}
            # A prefix gallery may exclude queries; require an explicit valid subset.
            if any(name not in positions for name in queries):
                raise ValueError("Same-collection query missing from selected gallery")
            query_indices = tuple(positions[name] for name in queries)
            paths = gallery
        else:
            query_indices = tuple(range(len(gallery), len(gallery) + len(queries)))
            paths = gallery + queries
        image_root = Path(image_root) if image_root is not None else root
        modalities = None
        if gallery_modality is not None or query_modality is not None:
            if not gallery_modality or not query_modality:
                raise ValueError("Both gallery and query modalities must be specified")
            if same_collection and gallery_modality != query_modality:
                raise ValueError(
                    "A shared collection cannot have conflicting modalities"
                )
            modalities = [gallery_modality] * len(gallery)
            if not same_collection:
                modalities += [query_modality] * len(queries)
        super().__init__(
            [image_root / path for path in paths], transform, modalities=modalities
        )
        self.eval_meta = {
            "query_indices": query_indices,
            "gallery_indices": tuple(range(len(gallery))),
            "ground_truth": tuple(gt),
            "exclude_self": same_collection,
        }
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.read_bytes())
        digest.update(repr((gallery_limit, query_limit, same_collection)).encode())
        self.metadata = {
            "split": split,
            "prefix": prefix,
            "annotations_sha256": digest.hexdigest(),
            "num_samples": len(self),
            "num_gallery": len(gallery),
            "num_queries": len(queries),
            "subset": gallery_limit is not None or query_limit is not None,
        }
        if modalities is not None:
            self.metadata["gallery_modality"] = gallery_modality
            self.metadata["query_modality"] = query_modality
