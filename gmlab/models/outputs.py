"""Explicit image-aligned output streams and spatial layout validation.

These functions do not execute models or access storage. Stream specifications
describe numerical outputs; input modality and retrieval role remain separate.
"""

from collections.abc import Mapping
import re

import torch


def validate_stream_specs(specifications: Mapping) -> dict:
    """Validate fixed per-image layouts before inference or opening writers.

    Each named stream declares layout (global/spatial), shape ([D]/[C,H,W]),
    branch, layer, normalization and whitening. Branch identifies the output
    selection, not the input modality. Shape must be known from the model/data
    configuration; variable spatial grids are not supported by this contract.
    """
    if not isinstance(specifications, Mapping) or not specifications:
        raise ValueError("Declare at least one output stream")
    required = {"layout", "shape", "branch", "layer", "normalization", "whitening"}
    result = {}
    for name, specification in specifications.items():
        if not isinstance(name, str) or not re.fullmatch("[a-z][a-z0-9_]*", name):
            raise ValueError("Output stream names must be lowercase identifiers")
        if not isinstance(specification, Mapping) or set(specification) != required:
            raise ValueError(f"Stream {name} requires exactly {sorted(required)}")
        value = dict(specification)
        rank = {"global": 1, "spatial": 3}.get(value["layout"])
        shape = value["shape"]
        if (
            rank is None
            or not isinstance(shape, (tuple, list))
            or len(shape) != rank
            or any(type(d) is not int or d < 1 for d in shape)
        ):
            raise ValueError(f"Stream {name} needs fixed [D] or [C,H,W] shape")
        for field in ("branch", "layer", "normalization"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"Stream {name} needs explicit {field}")
        if type(value["whitening"]) is not bool:
            raise ValueError(f"Stream {name} whitening must be boolean")
        value["shape"] = list(shape)
        result[name] = value
    return result


def align_output_streams(outputs, specifications: Mapping, indices: torch.Tensor):
    """Return stream -> {features, indices}, retaining the original row order.

    A model/adapter returns a mapping of named tensors [B,D] or [B,C,H,W].
    A tensor alone is allowed for a single declared stream. Each stream has B
    rows aligned to the same unique, nonnegative int64 input indices [B]. This
    rejects the ambiguous 2B-row tower concatenation. No reshaping, casting,
    normalization or input/output branch inference is performed.
    """
    specs = validate_stream_specs(specifications)
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.long
        or indices.ndim != 1
        or not len(indices)
        or (indices < 0).any()
        or indices.unique().numel() != indices.numel()
    ):
        raise ValueError("Input indices must be unique nonnegative int64 [B]")
    if isinstance(outputs, torch.Tensor) and len(specs) == 1:
        outputs = {next(iter(specs)): outputs}
    if not isinstance(outputs, Mapping) or set(outputs) != set(specs):
        raise ValueError("Model outputs must match the declared output streams")
    result = {}
    for name, spec in specs.items():
        values = outputs[name]
        if (
            not isinstance(values, torch.Tensor)
            or not values.is_floating_point()
            or tuple(values.shape) != (len(indices), *spec["shape"])
            or not torch.isfinite(values).all()
        ):
            raise ValueError(
                f"Stream {name} must be finite aligned [B,{spec['shape']}]"
            )
        result[name] = {"features": values, "indices": indices}
    return result


def tokens_to_spatial_map(tokens, *, grid, prefix_tokens):
    """Convert row-major [B,T,C] tokens to [B,C,H,W] using an explicit grid.

    The model adapter supplies the actual (H,W) and number of leading special
    tokens. Exactly H*W patch tokens must remain. No square-grid inference or
    padding is allowed; this operation does not normalize or sample patches.
    """
    if (
        not isinstance(grid, (list, tuple))
        or len(grid) != 2
        or any(type(d) is not int or d < 1 for d in grid)
        or type(prefix_tokens) is not int
        or prefix_tokens < 0
    ):
        raise ValueError("Declare positive (H,W) and nonnegative prefix_tokens")
    height, width = grid
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 3
        or tokens.shape[1] != height * width + prefix_tokens
        or tokens.shape[2] < 1
    ):
        raise ValueError("Token count differs from the explicit spatial grid")
    patches = tokens[:, prefix_tokens:]
    return patches.transpose(1, 2).reshape(len(tokens), tokens.shape[2], height, width)


def select_feature_streams(output: Mapping, batch_info: Mapping) -> dict:
    """Select declared named tensors without changing rows or computing heads.

    Models may expose additional numerical outputs; common extraction supplies
    output_streams in batch_info. A missing stream is a configuration error.
    """
    if not isinstance(output, Mapping):
        raise ValueError("Named stream selection requires a model output mapping")
    names = batch_info["output_streams"]
    missing = set(names) - output.keys()
    if missing:
        raise ValueError(f"Model output is missing feature streams: {sorted(missing)}")
    return {name: output[name] for name in names}
