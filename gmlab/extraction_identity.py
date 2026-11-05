"""Feature calculation descriptions and source compatibility; no publication IO.

Dataset-owned row/preprocessing declarations live in datasets.utils.feature_inputs.
This module binds model state, output semantics and precision to those declarations.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .datasets.utils.feature_inputs import describe_preprocessing
from .models.identity import model_digest


def _model_modules(model):
    modules = []
    for name, module in model.named_modules():
        class_id = f"{type(module).__module__}.{type(module).__qualname__}"
        # Components may retain a verified calculation ID across source moves.
        # Actual configuration and state are still included in the identity.
        class_id = getattr(type(module), "calculation_id", class_id)
        if not isinstance(class_id, str) or not class_id:
            raise ValueError("calculation_id must be a nonempty string")
        scalars = {
            key: value
            for key, value in vars(module).items()
            if not key.startswith("_")
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        modules.append(
            {
                "name": name,
                "class": class_id,
                "parameters": scalars,
            }
        )
    return modules


def _parents(references):
    result = []
    for reference in references:
        key = reference.get("artifact_key")
        payloads = reference.get("payloads")
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(c not in "0123456789abcdef" for c in key)
            or not payloads
        ):
            raise ValueError(
                "Parents require a full verified key and payload checksums"
            )
        normalized = []
        for payload in payloads:
            checksum = payload.get("sha256", "")
            if len(checksum) != 64 or any(
                c not in "0123456789abcdef" for c in checksum
            ):
                raise ValueError("Invalid parent payload checksum")
            normalized.append(
                {
                    name: payload[name]
                    for name in ("path", "sha256", "bytes", "format", "shape", "dtype")
                    if name in payload
                }
            )
        result.append({"artifact_key": key, "payloads": normalized})
    return result


def describe_adapter(adapter):
    """Describe a callable and its explicitly declared inference configuration."""
    if adapter is None:
        return None
    name = f"{adapter.__module__}.{adapter.__qualname__}"
    describe = getattr(getattr(adapter, "__self__", None), "inference_identity", None)
    if callable(describe):
        owner = adapter.__self__
        return {
            "callable": name,
            "owner": f"{type(owner).__module__}.{type(owner).__qualname__}",
            "configuration": describe(),
        }
    return name


def _feature_calculation(
    *,
    model,
    model_config,
    preprocessing,
    embedding_adapter=None,
    batch_adapter=None,
    output_space=None,
    storage_dtype="float32",
    parents=(),
    seed=0,
    realization="deterministic_seed_v1",
    algorithm_version="global_fp32_model_adapter_v1",
    compute=None,
):
    """Build the shared producer/consumer calculation without touching inputs."""
    from .training.adapters import image_batch

    adapter = describe_adapter(embedding_adapter)
    calculation = {
        "algorithm": algorithm_version,
        "compute": compute,
        "model": model_config,
        "model_modules": _model_modules(model),
        "state_sha256": model_digest(model),
        "adapter": adapter,
        "preprocessing": preprocessing,
        "output": output_space or {"layout": "global", "adapter": adapter},
        "storage": {
            "format": "gmlab_global_h5_v1",
            "dtype": str(np.dtype(storage_dtype)),
        },
        "parents": _parents(parents),
        "realization": realization,
        "seed": seed,
    }
    if batch_adapter is not None and batch_adapter is not image_batch:
        calculation["input_adapter"] = describe_adapter(batch_adapter)
    return calculation


def _stream_calculation(calculation, name, specification):
    return {
        **calculation,
        "output": {"stream": name, **specification},
        "storage": {
            **calculation["storage"],
            "format": (
                "gmlab_global_h5_v1"
                if specification["layout"] == "global"
                else "gmlab_spatial_h5_v1"
            ),
        },
    }


def validate_feature_source(
    recipe: Mapping[str, Any],
    expected_calculation: Mapping[str, Any],
    *,
    model: torch.nn.Module,
) -> dict[str, str | bool]:
    """Validate a source and return its executable inference precision arguments.

    recipe uses run_extraction's model/transform/adapter/output/streams/seed
    declarations. No dataset is constructed, input read, weights loaded or model
    forward executed. Model training flags are restored after describing eval
    behavior. Dataset selection, paths, devices and loader settings are excluded;
    state, preprocessing, adapters, output semantics and seed remain strict.
    For named outputs the expected calculation identifies the declared stream.

    Return a fresh dict with precision, matmul_tf32 and cudnn_tf32. Unspecified
    or None-valued recipe fields inherit only the verified source calculation;
    explicit conflicts are rejected. Copy the result into the recipe BEFORE
    preparing/exporting a reconstruction so deployment defaults cannot change
    its checked policy. No recipe, calculation or global flags are mutated.
    Storage dtype is independent and is not part of this returned binding.
    """
    from .components import resolve_component
    from .datasets.utils.transforms import build_image_pipelines
    from .datasets.utils.feature_description import (
        validate_feature_space,
        recorded_compute,
    )
    from .models.outputs import validate_stream_specs
    from .pipeline import resolve_pipeline_route

    pipeline = resolve_pipeline_route(recipe)
    if pipeline is not None:
        if recipe.get("adapter") is not None or recipe.get("input_adapter") is not None:
            raise ValueError("Pipeline cannot be mixed with extraction adapters")
        pipeline.validate_model(model, inference=True)

    def adapter(key):
        if pipeline is not None:
            if key == "input_adapter":
                return pipeline.prepare_inference
            return (
                pipeline.select_streams
                if recipe.get("streams")
                else pipeline.select_inference
            )
        value = recipe.get(key)
        return resolve_component(value) if isinstance(value, str) else value

    training = {module: module.training for module in model.modules()}
    try:
        model.eval()
        expected_compute = recorded_compute(expected_calculation)
        if expected_compute is None:
            raise ValueError(
                "Unknown source calculation.compute; precision cannot be inferred from storage dtype"
            )
        requested_precision = recipe.get("precision")
        if requested_precision is None:
            requested_precision = expected_compute["precision"]
        requested_precision = (
            "32-true" if requested_precision in (32, "32") else requested_precision
        )
        compute = {**expected_compute, "precision": requested_precision}
        for field in ("matmul_tf32", "cudnn_tf32"):
            if recipe.get(field) is not None:
                compute[field] = recipe[field]
        compute["float32_matmul_precision"] = (
            "high" if compute["matmul_tf32"] else "highest"
        )
        calculation = _feature_calculation(
            model=model,
            compute=compute,
            storage_dtype=recipe.get("storage_dtype") or "float32",
            model_config=getattr(model, "feature_config", recipe["model"]),
            preprocessing=describe_preprocessing(
                build_image_pipelines(recipe["transform"])
            ),
            embedding_adapter=adapter("adapter"),
            batch_adapter=adapter("input_adapter"),
            output_space=None if recipe.get("streams") else recipe["output"],
            seed=recipe.get("seed", 0),
        )
        if recipe.get("streams") is not None:
            specs = validate_stream_specs(recipe["streams"])
            name = expected_calculation.get("output", {}).get("stream")
            if name not in specs:
                raise ValueError(f"calculation.output.stream is not declared: {name}")
            calculation = _stream_calculation(calculation, name, specs[name])
        validate_feature_space(calculation, expected_calculation)
        return {key: compute[key] for key in ("precision", "matmul_tf32", "cudnn_tf32")}
    finally:
        for module, was_training in training.items():
            module.training = was_training
