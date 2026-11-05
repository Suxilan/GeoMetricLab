"""Shared native DataLoader construction and exact extraction row selection."""

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ...components import build_component, resolve_component
from .sampling import build_sampler


def build_loader(
    spec: Mapping[str, Any],
    transforms: Mapping | None = None,
    *,
    seed: int | None = None,
    dataset_arguments: Mapping | None = None,
) -> tuple[Dataset, DataLoader]:
    """Return (dataset, loader) from one split's reader/transform/loader recipe.

    The reader receives a transform only when explicitly configured and non-null.
    Sampler/batch_sampler receives that dataset; custom collate is retained.
    Index/sampling ownership remains with the Dataset/PyTorch sampler, never
    this constructor.
    Direct collate paths and invalid value types are checked before Dataset setup.
    """
    options = dict(spec.get("loader", {}))
    for name in ("collate_fn", "worker_init_fn"):
        if name not in options:
            continue
        value = options[name]
        if isinstance(value, str):
            value = resolve_component(value)
        elif (
            value is not None and not isinstance(value, Mapping) and not callable(value)
        ):
            raise TypeError(
                f"loader.{name} must be a callable path, component mapping, callable or null"
            )
        options[name] = value

    preprocessing = dict(dataset_arguments or {})
    if spec.get("transform") is not None:
        if "transform" in preprocessing:
            raise ValueError(
                "Dataset transform is declared by both split and injected arguments"
            )
        from .transforms import build_image_pipelines

        preprocessing["transform"] = build_image_pipelines(
            spec["transform"], transforms
        )
    dataset = build_component(spec["dataset"], **preprocessing)
    if seed is not None:
        # DataLoader iterator seeds must not consume model/dropout RNG. Worker
        # cursor recovery is a separate contract, even with this RNG separation.
        options.setdefault("generator", torch.Generator().manual_seed(seed))
    for name in ("collate_fn", "worker_init_fn"):
        if isinstance(options.get(name), Mapping):
            # Preserve factory construction order relative to Dataset setup.
            value = build_component(options[name])
            if value is not None and not callable(value):
                raise TypeError(f"loader.{name} must resolve to a callable or null")
            options[name] = value
    if isinstance(options.get("generator"), Mapping):
        options["generator"] = build_component(options["generator"])
    for key in ("sampler", "batch_sampler"):
        if key in options:
            specification = options[key]
            if (
                seed is not None
                and isinstance(specification, Mapping)
                and "seed" in specification
                and specification["seed"] is None
            ):
                specification = {**specification, "seed": seed}
            try:
                options[key] = build_sampler(specification, dataset)
            except (TypeError, ValueError, ImportError, AttributeError) as error:
                raise ValueError(f"Invalid loader.{key}: {error}") from error
    loader = DataLoader(dataset, **options)
    return dataset, loader


def selected_loader(loader: DataLoader, rows: Sequence[int]) -> DataLoader:
    """Replace sequential row selection, preserving native loader execution options.

    Extraction supplies exact unpadded rank rows, so sampler/batch_sampler and
    drop_last are deliberately replaced. Custom DataLoader subclasses cannot be
    reconstructed safely and are rejected rather than losing their behavior.
    """
    if type(loader) is not DataLoader:
        raise TypeError("Extraction requires a native DataLoader for row selection")
    managed = {
        "dataset",
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    }
    options = {}
    for name in inspect.signature(DataLoader).parameters.keys() - managed:
        if not hasattr(loader, name):
            raise ValueError(f"Cannot preserve DataLoader option {name}")
        options[name] = getattr(loader, name)
    return DataLoader(
        Subset(loader.dataset, rows), batch_size=loader.batch_size, **options
    )
