"""Image preprocessing built from explicit torchvision operations."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torchvision import transforms
from torchvision.transforms import v2

from ...components import resolve_component


def resolve_image_pipelines(specification, named_pipelines=None):
    """Resolve one pipeline or an explicit modality-to-pipeline mapping.

    Names refer to leaf pipelines, not recursive aliases. Returned definitions
    are not mutated; Dataset constructors receive callables after assembly.
    """
    named_pipelines = named_pipelines or {}

    def resolve(value):
        if isinstance(value, str):
            if value not in named_pipelines:
                raise ValueError(f"Unknown named image pipeline: {value}")
            return named_pipelines[value]
        return value

    if isinstance(specification, Mapping) and "by_modality" in specification:
        if set(specification) != {"by_modality"}:
            raise ValueError("by_modality cannot be mixed with pipeline operations")
        pipelines = specification["by_modality"]
        if not isinstance(pipelines, Mapping) or not pipelines:
            raise ValueError("by_modality requires a nonempty modality mapping")
        if any(not isinstance(name, str) or not name for name in pipelines):
            raise ValueError("Modality names must be nonempty strings")
        return {name: resolve(value) for name, value in pipelines.items()}
    return resolve(specification)


def build_image_pipelines(specification, named_pipelines=None):
    """Build one callable or explicitly selected per-modality callables."""
    pipelines = resolve_image_pipelines(specification, named_pipelines)
    if isinstance(specification, Mapping) and "by_modality" in specification:
        return {name: image_transform(value) for name, value in pipelines.items()}
    return image_transform(pipelines)


def image_transform(options: Mapping[str, Any] | Sequence[Mapping]) -> Callable:
    """Build a PIL RGB -> Tensor pipeline without training/runtime dependencies.

    New experiments use an ordered list of torchvision v2 operations. Resize
    interpolation and dtype are explicit YAML strings. External callable types
    can be supplied through the same component construction boundary.
    The earlier size/mean/std mapping preserves its v1 pixel behavior for Python
    callers; it is not a source of defaults for the explicit v2 pipeline.
    """
    if isinstance(options, Mapping):
        values = dict(options)
        operations = [
            transforms.Resize(tuple(values.pop("size"))),
            transforms.ToTensor(),
        ]
        mean, std = values.pop("mean", None), values.pop("std", None)
        if (mean is None) != (std is None) or values:
            raise ValueError("Transform accepts size, and optional mean/std together")
        if mean is not None:
            operations.append(transforms.Normalize(mean, std))
        return transforms.Compose(operations)
    operations = []
    for specification in options:
        constructor, arguments = _operation_definition(specification)
        operations.append(constructor(**arguments))
    if not operations:
        raise ValueError("An image pipeline must contain at least one operation")
    return v2.Compose(operations)


def _operation_definition(specification: Mapping) -> tuple[Callable, dict]:
    """Resolve one transform and normalize its YAML enum/dtype arguments."""
    if not isinstance(specification, Mapping):
        raise TypeError("Each transform operation must be a mapping with a type")
    if not isinstance(specification.get("type"), str) or not specification["type"]:
        raise ValueError("Each transform operation requires a nonempty type name")
    values = dict(specification)
    name = values.pop("type")
    constructor = getattr(v2, name, None)
    if "interpolation" in values:
        values["interpolation"] = transforms.InterpolationMode(values["interpolation"])
    if "dtype" in values:
        dtype = values["dtype"]
        if dtype not in {"float32", "float64", "float16", "bfloat16", "uint8"}:
            raise ValueError(f"Unsupported image dtype: {dtype}")
        values["dtype"] = getattr(torch, dtype)
    if constructor is None:
        constructor = resolve_component(name)
    return constructor, values


def validate_image_transform(options: Mapping | Sequence[Mapping]) -> None:
    """Check preprocessing arguments without reading images or model weights."""
    if isinstance(options, Mapping):
        image_transform(options)
        return
    if not options:
        raise ValueError("An image pipeline must contain at least one operation")
    for index, specification in enumerate(options):
        constructor, arguments = _operation_definition(specification)
        try:
            inspect.signature(constructor).bind(**arguments)
        except TypeError as error:
            raise ValueError(
                f"operation {index} ({specification['type']}): {error}"
            ) from error
