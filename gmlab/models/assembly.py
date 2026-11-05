"""Validated affine mounting and reconstruction, separate from model execution.

Source identity and cache compatibility are checked by extraction consumers.
This module checks the numerical mounting point and prepares self-contained
model recipes; it does not read cache files, fit parameters or load weights.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn

from ..components import component_definition


def build_affine(
    in_features: int, out_features: int, *, trainable: bool, bias: bool = True
) -> nn.Linear:
    """Construct a native Linear with explicit parameter trainability."""
    if type(trainable) is not bool:
        raise ValueError("Affine trainable must be boolean")
    return nn.Linear(in_features, out_features, bias=bias).requires_grad_(trainable)


def _mounting_layout(model: nn.Module, outputs) -> tuple[str, dict[str, int], object]:
    """Describe only known forwards, retaining legacy checkpoint registration."""
    from .geoencoder import (
        GeoEncoder,
        build_geoencoder,
    )
    from .branches import NamedBranches, build_named_branches

    if type(model) is GeoEncoder and "forward" not in vars(model):
        if type(model.whitening_layer) is not nn.Identity or model.final_norm:
            raise ValueError(
                "Whitening requires an unwhitened GeoEncoder before final L2"
            )
        if model.out_channels != model.aggregator.out_channels:
            raise ValueError(
                "GeoEncoder descriptor dimension differs from its aggregator"
            )
        return "single", {"descriptor": model.out_channels}, build_geoencoder
    if type(model) is NamedBranches and "forward" not in vars(model):
        if model.modality_routes:
            raise ValueError(
                "Routed descriptors require per-route source provenance; "
                "named-stream whitening sources cannot describe merged modality rows"
            )
        dimensions = {}
        for name, branch in model.branches.items():
            if name not in outputs:
                continue
            head = model.heads[branch.head]
            dimension = branch.dimension
            if dimension is None and model.head_adapters.get(branch.head) is None:
                dimension = getattr(head, "out_channels", None)
            if type(dimension) is not int or dimension < 1:
                raise ValueError(
                    f"Branch {name} needs an explicit descriptor dimension"
                )
            if (
                branch.dimension is not None
                and model.head_adapters.get(branch.head) is None
                and getattr(head, "out_channels", dimension) != dimension
            ):
                raise ValueError(f"Branch {name} descriptor dimensions disagree")
            dimensions[name] = dimension
        return "named", dimensions, build_named_branches
    raise ValueError("Whitening requires a native GeoEncoder or NamedBranches forward")


def _native_pipeline_adapters(kind, calculation):
    """Recognize the exact native selectors supported by affine mounting.

    A downstream pipeline may change feature mathematics; its name alone does
    not establish a safe mounting point. Only the unchanged native selectors
    are accepted here, alongside the existing explicit adapter contract.
    """
    from ..extraction_identity import describe_adapter
    from ..pipeline import GeoPipeline

    recorded = calculation.get("adapter")
    if not isinstance(recorded, Mapping):
        return False
    configuration = recorded.get("configuration", {})
    if not isinstance(configuration, Mapping):
        return False
    pipeline = GeoPipeline(retrieval_output=configuration.get("retrieval_output"))
    selected = (
        pipeline.select_inference if kind == "single" else pipeline.select_streams
    )
    return recorded == describe_adapter(selected) and calculation.get(
        "input_adapter"
    ) == describe_adapter(pipeline.prepare_inference)


def _validate_plan(model, calculations, assignments, *, trainable, final_norm):
    if type(trainable) is not bool or type(final_norm) is not bool:
        raise ValueError("Whitening trainable and final_norm must be boolean")
    if not isinstance(calculations, Mapping) or not calculations:
        raise ValueError("Provide named source calculations")
    if not isinstance(assignments, Mapping) or set(assignments) != set(calculations):
        raise ValueError("Each selected output needs exactly one affine assignment")
    for name in assignments.values():
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError(
                "Affine names must be nonempty module identifiers without '.'"
            )
    kind, dimensions, factory = _mounting_layout(model, calculations)
    registration_target = model.postprocessors if kind == "named" else nn.ModuleDict()
    for name in assignments.values():
        # ModuleDict uses Module.add_module: existing non-module attributes
        # (including training, forward and update) are not legal child names.
        # Check the actual target before creating or registering any affine.
        if hasattr(registration_target, name) and name not in registration_target:
            raise ValueError(f"Affine name {name!r} conflicts with a module attribute")
    if not set(calculations) <= dimensions.keys():
        raise ValueError("Unknown whitening output selection")
    if kind != "named" and (
        set(calculations) != set(dimensions) or len(set(assignments.values())) != 1
    ):
        raise ValueError(
            "Legacy encoders require all descriptor outputs to share one affine"
        )
    group_dimensions = {}
    for name, calculation in calculations.items():
        dimension = dimensions[name]
        group = assignments[name]
        if group in group_dimensions and group_dimensions[group] != dimension:
            raise ValueError("Shared affine inputs need the same descriptor dimension")
        group_dimensions[group] = dimension
        adapter = (
            "gmlab.training.adapters.descriptor_embeddings"
            if kind == "single"
            else "gmlab.models.outputs.select_feature_streams"
        )
        legacy_adapters = (
            calculation.get("adapter") == adapter
            and calculation.get("input_adapter") is None
        )
        if not legacy_adapters and not _native_pipeline_adapters(kind, calculation):
            raise ValueError(f"Unsupported {name} calculation.adapter or input_adapter")
        output = calculation.get("output", {})
        expected = {
            "layout": "global",
            "layer": "descriptor",
            "normalization": "none",
            "whitening": False,
        }
        if kind == "single":
            if output.get("branch") not in (None, "descriptor"):
                raise ValueError(
                    "Unsupported whitening source calculation: output.branch"
                )
            if output.get("stream") is not None:
                raise ValueError(
                    "Unsupported whitening source calculation: output.stream"
                )
            if output.keys() - expected.keys() - {"shape", "branch", "stream"}:
                raise ValueError("Unsupported whitening source calculation fields")
            if "shape" in output and output["shape"] != [dimension]:
                raise ValueError("Unsupported whitening source calculation shape")
            different = [
                f"output.{key}"
                for key, value in expected.items()
                if output.get(key) != value
            ]
            if different:
                raise ValueError(
                    "Unsupported whitening source calculation: " + ", ".join(different)
                )
        else:
            expected.update(stream=name, branch=name, shape=[dimension])
            if output != expected:
                location = "post-descriptor"
                raise ValueError(f"Unsupported {name} {location} output declaration")
        if kind == "named":
            branch = model.branches[name]
            if branch.normalize or (
                branch.postprocess is not None
                and type(model.postprocessors[branch.postprocess]) is not nn.Identity
            ):
                raise ValueError(f"Branch {name} must be unwhitened before final L2")
    if kind == "named":
        for group in group_dimensions:
            if group in model.postprocessors:
                raise ValueError(
                    f"Affine name {group!r} conflicts with an existing postprocessor"
                )
    return kind, dimensions, group_dimensions, factory


def _output_dimensions(dimension, groups):
    dimensions = (
        dict.fromkeys(groups, dimension) if type(dimension) is int else dict(dimension)
    )
    if set(dimensions) != set(groups):
        raise ValueError("Whitening dimensions must cover exactly the assigned affines")
    for name, value in dimensions.items():
        if type(value) is not int or not 0 < value <= groups[name]:
            raise ValueError(
                "Whitening dimension must be positive and <= descriptor dimension"
            )
    return dimensions


def prepare_whitening(
    recipe: Mapping,
    calculations: Mapping[str, Mapping],
    *,
    model: nn.Module,
    dimension: int | Mapping[str, int],
    assignments: Mapping[str, str],
    trainable: bool,
    final_norm: bool = True,
) -> dict:
    """Prepare a reconstruction recipe without mutating the model or source recipe.

    assignments maps each selected descriptor output to an affine name. Reusing
    a name explicitly shares one affine; different names remain independent.
    dimension is a common output width or affine-name -> width. Source cache
    identity/precision must already have been verified by the caller. The result
    retains that binding, omits initialization and loads its complete state later.
    """
    kind, _, groups, factory = _validate_plan(
        model, calculations, assignments, trainable=trainable, final_norm=final_norm
    )
    dimensions = _output_dimensions(dimension, groups)
    constructor, arguments = component_definition(recipe["model"])
    if constructor is not factory:
        raise ValueError(
            "Whitening reconstruction requires the native model factory recipe"
        )
    result = deepcopy(recipe)
    result.pop("initialization", None)
    if kind == "single":
        if recipe.get("streams") is not None:
            raise ValueError(
                "Named stream whitening assembly is not supported by GeoEncoder"
            )
        if recipe["output"] != calculations["descriptor"]["output"]:
            raise ValueError("Recipe output differs from its source calculation")
    else:
        streams = recipe.get("streams", {})
        for name, calculation in calculations.items():
            if (
                name not in streams
                or {"stream": name, **streams[name]} != calculation["output"]
            ):
                raise ValueError(
                    f"Recipe stream {name} differs from its source calculation"
                )
    if kind != "named":
        result["model"] = {
            "type": "GeoEncoder",
            **deepcopy(arguments),
            "whitening": True,
            "whitening_dim": next(iter(dimensions.values())),
            "whitening_trainable": trainable,
            "final_norm": final_norm,
        }
    else:
        model_recipe = {"type": "NamedBranches", **deepcopy(arguments)}
        processors = model_recipe.setdefault("postprocessors", {})
        for name, width in dimensions.items():
            if name in processors:
                raise ValueError(
                    f"Affine name {name!r} conflicts with recipe postprocessor"
                )
            processors[name] = {
                "type": "Affine",
                "in_features": groups[name],
                "out_features": width,
                "trainable": trainable,
            }
        for output, affine in assignments.items():
            branch = model_recipe["branches"][output]
            model_recipe["branches"][output] = {
                **branch,
                "postprocess": affine,
                "normalize": final_norm,
            }
        result["model"] = model_recipe
    for output, affine in assignments.items():
        target = result["output"] if kind == "single" else result["streams"][output]
        target.update(whitening=True, normalization="l2" if final_norm else "none")
        if kind != "single":
            target["layer"] = "whitening"
        if "shape" in target:
            target["shape"] = [dimensions[affine]]
    return result


def _affine_layer(model, parameters, input_dimension, *, trainable):
    if not isinstance(parameters, Mapping) or set(parameters) != {"weight", "bias"}:
        raise ValueError("Expected whitening weight and bias tensors")
    weight, bias = parameters["weight"], parameters["bias"]
    if (
        not isinstance(weight, torch.Tensor)
        or not isinstance(bias, torch.Tensor)
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or weight.ndim != 2
        or weight.shape[1] != input_dimension
        or not 0 < weight.shape[0] <= input_dimension
        or bias.shape != (weight.shape[0],)
        or not torch.isfinite(weight).all()
        or not torch.isfinite(bias).all()
    ):
        raise ValueError("Expected finite FP32 whitening weight [d,D] and bias [d]")
    reference = next(model.parameters(), weight)
    if reference.dtype != torch.float32:
        raise ValueError("Whitening assembly requires an FP32 source model")
    layer = build_affine(input_dimension, len(bias), trainable=trainable).to(
        reference.device
    )
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    return layer.train(model.training)


def attach_whitening(
    model: nn.Module,
    calculations: Mapping[str, Mapping],
    parameters: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    assignments: Mapping[str, str],
    trainable: bool,
    final_norm: bool = True,
) -> nn.Module:
    """Mount validated FP32 affines after descriptor normalization, before final L2.

    parameters maps affine names to finite weight [d,D] and bias [d]. All source,
    shape and sharing checks complete before mutation. No cache IO, fitting or
    model forward occurs. Selected outputs share a parameter object only when
    their assignments refer to the same affine name; unselected outputs remain.
    """
    kind, _, groups, _ = _validate_plan(
        model, calculations, assignments, trainable=trainable, final_norm=final_norm
    )
    if set(parameters) != set(groups):
        raise ValueError("Whitening parameters must cover exactly the assigned affines")
    layers = {
        name: _affine_layer(model, parameters[name], width, trainable=trainable)
        for name, width in groups.items()
    }
    if kind != "named":
        layer = next(iter(layers.values()))
        model.whitening_layer = layer
        model.out_channels = layer.out_features
        model.final_norm = final_norm
    else:
        for name, layer in layers.items():
            model.postprocessors[name] = layer
        for output, affine in assignments.items():
            model.branches[output] = replace(
                model.branches[output], postprocess=affine, normalize=final_norm
            )
    return model
