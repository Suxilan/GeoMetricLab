"""Shared, read-only result provenance for training and independent evaluation."""

import copy
import inspect
import json
import platform
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch

from ..artifacts import identity_key
from ..components import component_definition
from ..evaluation.retrieval import evaluate_retrieval
from ..utils.provenance import code_version, dependency_versions

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from torch.utils.data import Dataset


def stable_digest(value: dict) -> str:
    """Hash JSON configuration by value; key order/whitespace are irrelevant.

    Reject non-JSON and non-finite values instead of silently stringifying them.
    This identifies configuration, not weights, annotation contents or code.
    """
    return identity_key(value)


def _retrieval_settings(
    dataset: "Dataset", options: Mapping[str, Any]
) -> dict[str, Any]:
    # Bind the same function/kwargs as the evaluator, so omitted defaults cannot
    # drift into a separately maintained definition of the calculation.
    bound = inspect.signature(evaluate_retrieval).bind_partial(
        **dataset.eval_meta, **options
    )
    bound.apply_defaults()
    return {
        name: bound.arguments[name]
        for name, parameter in inspect.signature(evaluate_retrieval).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _model_settings(component: Mapping[str, Any]) -> dict[str, Any]:
    component = dict(component)
    component.pop("freeze", None)
    constructor, parameters = component_definition(component)
    bound = inspect.signature(constructor).bind_partial(**parameters)
    bound.apply_defaults()
    resolved = {}
    defaults_from_code = []
    for name, value in bound.arguments.items():
        try:
            resolved[name] = json.loads(json.dumps(value, allow_nan=False))
        except TypeError:
            # A custom component may default to a Python callable/object. Keep
            # its configured import/spec and name the dependency on pinned code;
            # never hash a memory-address repr as though it were stable config.
            defaults_from_code.append(name)
    return {
        "component": component,
        "resolved_parameters": resolved,
        "defaults_from_code": defaults_from_code,
    }


def evaluation_provenance(
    config: Mapping[str, Any],
    split: str,
    dataset: "Dataset",
    model: torch.nn.Module,
    trainer: "pl.Trainer",
    *,
    embedding_adapter: str | Mapping,
    extract: bool = False,
    inference_execution: Mapping | None = None,
) -> dict[str, Any]:
    """Snapshot the calculation actually selected by evaluate/extract.

    embedding_adapter is the actual import path selected by the entrypoint,
    not a second interpretation of configuration. Only the selected split is
    saved: model construction/freeze policy, data reader/loader, preprocessing
    specification and realized transforms, embedding
    adapter, Trainer options and fully bound retrieval settings. Extraction has
    retrieval=None because no distance/ranking is executed. Annotation metadata
    has its own digest; computation.sha256 never claims to identify a complete
    benchmark. Code/dependencies/environment are linked alongside that snapshot.
    """
    from ..pipeline import resolve_pipeline_route
    from ..extraction_identity import describe_adapter

    pipeline = resolve_pipeline_route(config)
    trainer_options = {
        **config.get("runtime", {}).get(
            "evaluation_trainer", {"accelerator": "cpu", "devices": 1}
        ),
        "logger": False,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    calculation = copy.deepcopy(
        {
            "operation": "extract" if extract else "evaluate",
            "split": split,
            "model": {
                **_model_settings(config["model"]),
                "freeze_prefixes": config["model"].get("freeze", []),
                "representation": repr(model),
            },
            "data": config["data"][split],
            "preprocessing": repr(dataset.transform),
            "embedding_adapter": embedding_adapter,
            "inference_batch_adapter": (
                describe_adapter(pipeline.prepare_inference)
                if pipeline is not None
                else config.get("adapters", {}).get("inference_batch", "image")
            ),
            **({"pipeline": config["pipeline"]} if "pipeline" in config else {}),
            "retrieval": (
                None if extract else _retrieval_settings(dataset, config["evaluation"])
            ),
            "trainer": trainer_options,
            "inference": {
                key: inference_execution[key]
                for key in (
                    "precision",
                    "matmul_tf32",
                    "cudnn_tf32",
                    "float32_matmul_precision",
                )
                if inference_execution is not None and key in inference_execution
            },
        }
    )
    calculation = json.loads(json.dumps(calculation, allow_nan=False))
    return {
        "data_metadata": copy.deepcopy(getattr(dataset, "metadata", None)),
        "computation": {"config": calculation, "sha256": stable_digest(calculation)},
        "code": code_version(),
        "dependencies": dependency_versions(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "world_size": trainer.world_size,
            "accelerator": type(trainer.accelerator).__name__,
            "strategy": type(trainer.strategy).__name__,
            "device": str(trainer.strategy.root_device),
            "precision": str(trainer.precision),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
