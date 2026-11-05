"""Declared extraction inputs: stable rows and deterministic preprocessing.

Readers own this contract. Extraction does not infer computation from class names.
"""

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
import inspect
import json

import torch
from torchvision import transforms
from torchvision.transforms import v2

from ...utils.io import canonical_json, file_digest
from .identities import integer_tensor


def describe_preprocessing(transform):
    """Describe supported deterministic RGB operations without executing them."""
    if isinstance(transform, Mapping):
        return {
            "by_modality": {
                name: describe_preprocessing(pipeline)
                for name, pipeline in sorted(transform.items())
            }
        }
    if type(transform) not in (transforms.Compose, v2.Compose):
        raise ValueError("Declare deterministic torchvision Compose preprocessing")
    operations = []
    for operation in transform.transforms:
        name = type(operation).__name__
        if type(operation) in (transforms.Resize, v2.Resize):
            values = {
                "size": operation.size,
                "max_size": operation.max_size,
                "interpolation": operation.interpolation.value,
                "antialias": operation.antialias,
            }
        elif type(operation) is transforms.ToTensor:
            values = {
                "uint8_scale": 255,
                "dtype": str(torch.get_default_dtype()),
                "layout": "CHW",
            }
        elif type(operation) is v2.ToImage:
            values = {"layout": "CHW", "scale": False}
        elif type(operation) is v2.ToDtype:
            if not isinstance(operation.dtype, torch.dtype):
                raise ValueError("Per-type ToDtype mappings are not supported")
            values = {"dtype": str(operation.dtype), "scale": operation.scale}
        elif type(operation) in (transforms.Normalize, v2.Normalize):
            values = {
                "mean": list(operation.mean),
                "std": list(operation.std),
                "inplace": operation.inplace,
            }
        elif type(operation) is v2.ToPureTensor:
            values = {}
        else:
            raise ValueError(
                f"Untracked preprocessing: {type(operation).__module__}.{name}"
            )
        operations.append({"type": f"{type(operation).__module__}.{name}", **values})
    return {
        "decode": "PIL.open.convert_RGB_no_exif_transpose",
        "operations": operations,
    }


def dataset_preprocessing(dataset):
    """Read the reader's explicit, JSON-serializable preprocessing declaration."""
    if not hasattr(dataset, "feature_preprocessing"):
        raise ValueError("Dataset must declare feature_preprocessing")
    return json.loads(canonical_json(dataset.feature_preprocessing))


def image_feature_samples(dataset, *, source, input_root):
    """Describe native RGB path rows, crops and occurrences without decoding."""
    root = Path(input_root).absolute()
    occurrences, samples = Counter(), []
    for row, locator in enumerate(dataset.paths):
        path = Path(locator).absolute()
        try:
            sample_id = path.relative_to(root).as_posix()
        except ValueError:
            sample_id = path.resolve().relative_to(root.resolve()).as_posix()
        occurrence = occurrences[sample_id]
        occurrences[sample_id] += 1
        samples.append(
            {
                "row_id": row,
                "source_id": source,
                "sample_id": sample_id,
                "view": (
                    dataset.modalities[row]
                    if getattr(dataset, "modalities", None) is not None
                    else "default"
                ),
                "region": (
                    dataset.regions[row] if getattr(dataset, "regions", None) else None
                ),
                "occurrence": occurrence,
                "content_sha256": file_digest(path),
                "locator": str(path),
            }
        )
    return samples


def dataset_samples(
    dataset, *, source: str, input_root: str | Path | None
) -> list[dict]:
    """Validate one declaration per manifest row, scoped to this source.

    feature_samples accepts source/input_root keywords. Existing independent
    readers may instead expose a no-argument method with a fixed source scope.
    """
    if not callable(getattr(dataset, "feature_samples", None)):
        raise ValueError("Dataset must declare feature_samples")
    method = dataset.feature_samples
    signature = inspect.signature(method)
    try:
        signature.bind(source=source, input_root=input_root)
    except TypeError:
        signature.bind()  # Existing explicit sources may bind their own scope.
        samples = method()
    else:
        samples = method(source=source, input_root=input_root)
    required = {
        "row_id",
        "source_id",
        "sample_id",
        "view",
        "region",
        "occurrence",
        "content_sha256",
    }
    if len(samples) != len(dataset) or not samples:
        raise ValueError("feature_samples must cover every stable input row")
    validated = []
    for i, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or not required <= sample.keys():
            raise ValueError("Invalid explicit feature sample identity fields")
        row = integer_tensor(sample["row_id"], name="feature row_id", minimum=0)
        occurrence = integer_tensor(
            sample["occurrence"], name="feature occurrence", minimum=0
        )
        digest = sample.get("content_sha256", "")
        if (
            row.ndim != 0
            or occurrence.ndim != 0
            or int(row) != i
            or sample["source_id"] != source
            or not isinstance(sample["sample_id"], str)
            or not sample["sample_id"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Invalid explicit feature sample identity")
        validated.append({**sample, "row_id": int(row), "occurrence": int(occurrence)})
    return validated
