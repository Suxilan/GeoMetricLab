"""Canonical sample rows and shared flat/grouped Tensor collation."""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch

from .identities import integer_tensor


def validate_modality(value: Any) -> str:
    """Return an explicit nonempty modality name, never a dataset-name code."""
    if not isinstance(value, str) or not value:
        raise ValueError("modality must be a nonempty string")
    return value


def validate_sample_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one input row while retaining independent physical identity.

    sample_id is (source scope, physical ID). Repeated exposures may share it;
    class labels and Dataset indices do not substitute for physical IDs. Extra
    annotations are preserved without inferring pairing, sequence or validity.
    """
    if not isinstance(row, Mapping) or "inputs" not in row:
        raise ValueError("A sample row requires inputs and explicit identity metadata")
    result = dict(row)
    result["modality"] = validate_modality(row.get("modality"))
    identity = row.get("sample_id")
    if not isinstance(identity, tuple) or len(identity) != 2:
        raise ValueError("sample_id must be a (source_scope, physical_id) tuple")
    scope, physical_id = identity
    if not isinstance(scope, str) or not scope:
        raise ValueError("sample_id source scope must be a nonempty string")
    if isinstance(physical_id, bool) or not isinstance(physical_id, (str, Integral)):
        raise ValueError("sample_id physical ID must be a string or integer")
    if isinstance(physical_id, str) and not physical_id:
        raise ValueError("sample_id physical ID must be nonempty")
    result["sample_id"] = (
        scope,
        int(physical_id) if isinstance(physical_id, Integral) else physical_id,
    )
    if "label" in row:
        label = integer_tensor(row["label"], name="label")
        if label.ndim:
            raise ValueError("label must be an integer scalar")
        result["label"] = int(label)
    return result


def _flatten_samples(batch: Sequence[Mapping[str, Any]]):
    if not batch:
        raise ValueError("Cannot collate an empty sample batch")
    rows, item_indices, item_positions = [], [], []
    for item_position, item in enumerate(batch):
        if not isinstance(item, Mapping) or "index" not in item or "rows" not in item:
            raise ValueError("Each item requires its Dataset index and rows")
        index = integer_tensor(item["index"], name="Dataset item index", minimum=0)
        if index.ndim:
            raise ValueError("Dataset item index must be scalar")
        if (
            not isinstance(item["rows"], Sequence)
            or isinstance(item["rows"], (str, bytes))
            or not item["rows"]
        ):
            raise ValueError("Each item must contain at least one sample row")
        for row in item["rows"]:
            rows.append(validate_sample_row(row))
            item_indices.append(int(index))
            item_positions.append(item_position)
    return rows, item_indices, item_positions


def _collate_samples(batch, *, grouped):
    rows, item_indices, item_positions = _flatten_samples(batch)
    modalities = [row["modality"] for row in rows]
    names = list(dict.fromkeys(modalities))
    positions = {
        name: torch.tensor(
            [i for i, value in enumerate(modalities) if value == name], dtype=torch.long
        )
        for name in names
    }
    inputs = [row["inputs"] for row in rows]
    if any(not torch.is_tensor(value) for value in inputs):
        raise ValueError("Collation requires transformed Tensor inputs [N,...]")
    if grouped:
        images = {}
        for name, indices in positions.items():
            values = [inputs[i] for i in indices.tolist()]
            if any(value.shape != values[0].shape for value in values):
                raise ValueError(f"Inconsistent Tensor shapes within modality {name}")
            images[name] = torch.stack(values)
    else:
        if any(value.shape != inputs[0].shape for value in inputs):
            raise ValueError(
                "Stacked sample inputs must have one shape; use group_samples"
            )
        images = torch.stack(inputs)
    result = {
        "images": images,
        "modalities": modalities,
        "modality_names": names,
        "modality_ids": torch.tensor(
            [names.index(name) for name in modalities], dtype=torch.long
        ),
        "positions": positions,
        "sample_ids": [row["sample_id"] for row in rows],
        "item_indices": torch.tensor(item_indices, dtype=torch.long),
        "item_positions": torch.tensor(item_positions, dtype=torch.long),
        "row_metadata": [
            {
                key: value
                for key, value in row.items()
                if key not in {"inputs", "sample_id", "modality", "label"}
            }
            for row in rows
        ],
    }
    labels = ["label" in row for row in rows]
    if any(labels) and not all(labels):
        raise ValueError("Labels must be present on every row or none")
    if all(labels):
        result["labels"] = integer_tensor([row["label"] for row in rows], name="labels")
    return result


def stack_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack canonical groups as [N,...], preserving row order and duplicates.

    modality_ids index the returned batch-local modality_names vocabulary. Stable
    modality names, physical sample_ids and Dataset item_indices remain separate.
    item_positions [N] identifies the batch item occurrence for each row, even
    when multiple sampled items have the same Dataset index.
    """
    return _collate_samples(batch, grouped=False)


def group_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Group canonical rows by modality; positions map back to original row order.

    Each modality may have a different Tensor shape/count. No cross-modality
    correspondence or temporal synchronization is inferred from row positions.
    """
    return _collate_samples(batch, grouped=True)


# Public collator output layout, consumed by training assembly.
stack_samples.batch_format = "stacked_id"
group_samples.batch_format = "modality_id"


def cross_modality_pairs(labels: torch.Tensor, modalities: Sequence[str]):
    """Build directed ID pairs across modalities in the supplied row order.

    These are static within-batch candidates; no embedding mining or communication
    occurs. Equal IDs form positives, different IDs form negatives.
    """
    if labels.ndim != 1 or len(labels) != len(modalities):
        raise ValueError("Labels and modalities must share row order")
    vocabulary = {name: index for index, name in enumerate(dict.fromkeys(modalities))}
    ids = torch.tensor([vocabulary[name] for name in modalities], device=labels.device)
    eligible = ids[:, None] != ids[None, :]
    same_class = labels[:, None] == labels[None, :]
    return (*torch.where(eligible & same_class), *torch.where(eligible & ~same_class))


def group_cross_modal_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Group canonical inputs and restrict static ID pairs to distinct modalities."""
    result = group_samples(batch)
    result["indices_tuple"] = cross_modality_pairs(
        result["labels"], result["modalities"]
    )
    return result


group_cross_modal_samples.batch_format = "modality_id"


def image_batch(batch: Mapping[str, Any]) -> tuple[torch.Tensor, dict]:
    """GT-free default-collated images [N,C,H,W] and stable indices [N].

    Optional sample_id columns follow PyTorch default_collate; grouped callers
    use modality_image_batch with already row-major sample_ids instead.
    """
    images, indices = batch["images"], batch["index"]
    if (
        indices.ndim != 1
        or len(images) != len(indices)
        or indices.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("Inference indices must be integer [N], aligned with images")
    metadata = {}
    if "sample_id" in batch:
        # default_collate transposes each (scope, physical_id) tuple into columns.
        scopes, physical_ids = batch["sample_id"]
        if len(scopes) != len(indices) or len(physical_ids) != len(indices):
            raise ValueError("Collated sample identity columns must align with images")
        metadata["sample_ids"] = list(zip(scopes, physical_ids))
    if "modality" in batch:
        metadata["modalities"] = batch["modality"]
    return images, {"index": indices, "num_images": len(indices), **metadata}
