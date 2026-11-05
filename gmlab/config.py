"""Read explicit YAML experiments without constructing datasets or models."""

import copy
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import yaml

from .components import component_definition, resolve_adapter, resolve_component


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    """Report malformed YAML sections at their configuration path."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _validate_component(
    specification: Any,
    location: str,
    *,
    reserved: tuple[str, ...] = (),
    injected: tuple[str, ...] = (),
) -> None:
    """Check declared constructor arguments without instantiating the component."""
    if not isinstance(specification, Mapping) or "type" not in specification:
        raise ValueError(f"{location} requires a type and explicit arguments")
    try:
        constructor = resolve_component(specification["type"])
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {location}: {error}") from error
    arguments = {
        name: value
        for name, value in specification.items()
        if name not in {"type", *reserved}
    }
    conflicts = arguments.keys() & set(injected)
    if conflicts:
        raise ValueError(
            f"{location} cannot override injected arguments: {sorted(conflicts)}"
        )
    arguments.update(dict.fromkeys(injected))
    try:
        inspect.signature(constructor).bind(**arguments)
    except TypeError as error:
        raise ValueError(f"Invalid {location}: {error}") from error


def _validate_data(data: Mapping[str, Any]) -> None:
    """Check reader and loader arguments without opening the data collection."""
    unknown = data.keys() - {"transforms", "train", "valid", "test"}
    if unknown:
        raise ValueError(f"Unknown data sections: {sorted(unknown)}")
    if "train" not in data:
        raise ValueError("data.train is required")

    splits = []
    for name in ("train", "valid", "test"):
        if name not in data:
            continue
        location = f"data.{name}"
        split = _require_mapping(data[name], location)
        if name == "train":
            try:
                sources = training_source_specs(split)
            except ValueError as error:
                raise ValueError(f"Invalid {location}: {error}") from error
            for source, specification in sources.items():
                source_location = (
                    location if source is None else f"{location}.sources.{source}"
                )
                splits.append((source_location, specification, False))
        else:
            try:
                sources = evaluation_sources(split)
            except ValueError as error:
                raise ValueError(f"Invalid {location}: {error}") from error
            for source, specification in sources.items():
                source_location = (
                    location if source is None else f"{location}.sources.{source}"
                )
                splits.append((source_location, specification, True))
    for location, split, evaluation in splits:
        _validate_split(location, split, evaluation, data)


def _validate_split(location, split, evaluation, data):
    """Validate one source without constructing readers or model state."""

    allowed = {"dataset", "transform", "loader", "pipeline"}
    if evaluation:
        allowed |= {"evaluation", "adapters"}
    else:
        allowed |= {"loss", "adapters"}
    unknown = split.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {location} options: {sorted(unknown)}")
    if "dataset" not in split:
        raise ValueError(f"{location} requires dataset")
    if not evaluation:
        if "loss" in split:
            _validate_loss(split["loss"], f"{location}.loss")
        adapters = _require_mapping(split.get("adapters", {}), f"{location}.adapters")
        if adapters.keys() - {"batch", "embedding"}:
            raise ValueError(f"Unknown {location}.adapters options")
        for name, adapter in adapters.items():
            try:
                resolve_adapter(adapter)
            except (ImportError, AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {location}.adapters.{name}: {error}"
                ) from error
    if evaluation:
        _require_mapping(split.get("evaluation", {}), f"{location}.evaluation")
        adapters = _require_mapping(split.get("adapters", {}), f"{location}.adapters")
        unknown_adapters = adapters.keys() - {
            "inference_batch",
            "inference_embedding",
        }
        if unknown_adapters:
            raise ValueError(f"Unknown {location}.adapters: {sorted(unknown_adapters)}")
        for adapter_name, adapter in adapters.items():
            try:
                resolve_adapter(adapter)
            except (ImportError, AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {location}.adapters.{adapter_name}: {error}"
                ) from error
    _validate_component(
        split["dataset"],
        f"{location}.dataset",
        injected=("transform",) if split.get("transform") is not None else (),
    )
    _validate_split_transform(location, split, data.get("transforms", {}))
    _validate_split_loader(location, split.get("loader", {}))


def _validate_split_transform(location, split, transforms):
    """Resolve and check image pipelines without reading samples."""
    if split.get("transform") is not None:
        from .datasets.utils.transforms import (
            resolve_image_pipelines,
            validate_image_transform,
        )

        transform = split["transform"]
        if isinstance(transform, str) and transform not in transforms:
            raise ValueError(
                f"Unknown image pipeline at {location}.transform: {transform}"
            )
        try:
            pipelines = resolve_image_pipelines(transform, transforms)
            if isinstance(transform, Mapping) and "by_modality" in transform:
                for modality, pipeline in pipelines.items():
                    try:
                        validate_image_transform(pipeline)
                    except (TypeError, ValueError, KeyError) as error:
                        raise ValueError(f"modality {modality}: {error}") from error
            else:
                validate_image_transform(pipelines)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"Invalid {location}.transform: {error}") from error


def _validate_split_loader(location, specification):
    """Check native loader arguments and explicitly configured extensions."""
    from torch.utils.data import DataLoader

    loader = _require_mapping(specification, f"{location}.loader")
    try:
        inspect.signature(DataLoader).bind(dataset=None, **loader)
    except TypeError as error:
        raise ValueError(f"Invalid {location}.loader: {error}") from error
    for sampler in ("sampler", "batch_sampler"):
        if sampler in loader:
            from .datasets.utils.sampling import validate_sampler

            try:
                validate_sampler(loader[sampler])
            except (TypeError, ValueError, ImportError, AttributeError) as error:
                raise ValueError(
                    f"Invalid {location}.loader.{sampler}: {error}"
                ) from error
    collate = loader.get("collate_fn")
    if isinstance(collate, Mapping):
        _validate_component(collate, f"{location}.loader.collate_fn")
    elif collate is not None:
        try:
            if isinstance(collate, str):
                resolve_component(collate)
            elif not callable(collate):
                raise TypeError(
                    "Expected a registered callable, import path, component mapping, "
                    "callable or null"
                )
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid {location}.loader.collate_fn: {error}"
            ) from error


def _merge(base: Mapping, overrides: Mapping) -> dict:
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_recipe(path: str | Path) -> dict[str, Any]:
    """Read one YAML file and its optional single relative ``extends`` base.

    Mappings merge recursively; lists and scalars replace earlier values.
    Paths inside the experiment remain relative to the launch working directory.
    Only extends is relative to its declaring file. No interpolation or implicit
    component discovery is performed.
    """
    return _read_recipe(Path(path).resolve(), frozenset())


def read_training_recipe(
    path: str | Path, *, resume: str | None = None, validate: bool = True
) -> dict[str, Any]:
    """Read a recipe and apply the resume override.

    Validation defaults to enabled. CLI development mode defers it until after
    applying its explicit runtime overrides, then validates the resulting recipe.
    """
    config = read_recipe(path)
    runtime = config.get("runtime")
    if isinstance(runtime, dict) and "experiment_name" in runtime:
        name = runtime["experiment_name"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise ValueError("runtime.experiment_name must be a single directory name")
        runtime.setdefault("run_dir", str(Path("results/gmlab") / name))
        logging = runtime.get("logging")
        if isinstance(logging, dict) and isinstance(logging.get("tracking"), dict):
            tracking = logging["tracking"]
            if tracking.get("experiment_name", name) != name:
                raise ValueError(
                    "SwanLab experiment_name must match runtime.experiment_name"
                )
            tracking.setdefault("experiment_name", name)
    if resume is not None:
        runtime = config.get("runtime")
        if not isinstance(runtime, Mapping):
            raise ValueError("runtime must be a mapping")
        config["runtime"] = {**runtime, "resume": resume}
    if validate:
        validate_training_config(config)
    return config


def _read_recipe(path: Path, parents: frozenset[Path]) -> dict[str, Any]:
    if path in parents:
        raise ValueError(f"Cyclic recipe inheritance: {path}")
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment must be a YAML mapping: {path}")
    parent = config.pop("extends", None)
    if parent is None:
        return config
    if not isinstance(parent, str):
        raise TypeError("extends must name one YAML file")
    base = _read_recipe((path.parent / parent).resolve(), parents | {path})
    return _merge(base, config)


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Reject unknown experiment sections before loading data or pretrained weights.

    Native Trainer/component constructors validate their own arguments. This
    boundary validates ownership and required sections, not a second copy of
    every third-party option or a recursively inferred component graph.
    """
    _require_mapping(config, "experiment")
    allowed = {
        "model",
        "loss",
        "optimizer",
        "scheduler",
        "data",
        "evaluation",
        "adapters",
        "pipeline",
        "runtime",
        "initialization",
    }
    unknown = config.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown experiment sections: {sorted(unknown)}")
    required = {"model", "optimizer", "data", "runtime"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing experiment sections: {sorted(missing)}")
    from .pipeline import validate_pipeline_config

    _require_mapping(config["data"], "data")
    validate_pipeline_config(config)
    for section in required | {"evaluation", "adapters", "initialization"}:
        if section in config:
            _require_mapping(config[section], section)
    _validate_runtime(config["runtime"], config.get("initialization", {}))
    _validate_data(config["data"])

    sources = training_source_specs(config["data"]["train"])
    if config.get("loss") is None and any(
        source.get("loss") is None for source in sources.values()
    ):
        raise ValueError("Each training source needs a loss or a global loss")
    if {"valid", "test"} & config["data"].keys() and "evaluation" not in config:
        raise ValueError("Retrieval loaders require evaluation settings")
    if "valid" not in config["data"]:
        runtime = config["runtime"]
        if (
            runtime.get("validate_before_fit")
            or runtime.get("final_validation", "scheduled") != "scheduled"
        ):
            raise ValueError("Validation policy requires data.valid")
        monitor = runtime["checkpoint"].get("monitor")
        if monitor and monitor.startswith("val/"):
            raise ValueError("A validation monitor requires data.valid")
    _validate_model_and_loss(config)
    _validate_optimization(config)
    adapters = config.get("adapters", {})
    allowed_adapters = {"batch", "embedding", "inference_batch", "inference_embedding"}
    if adapters.keys() - allowed_adapters:
        raise ValueError(
            f"Unknown adapters: {sorted(adapters.keys() - allowed_adapters)}"
        )

    for name, adapter in adapters.items():
        try:
            resolve_adapter(adapter)
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid adapters.{name}: {error}") from error


def _validate_runtime(runtime: Mapping, initialization: Mapping) -> None:
    """Validate execution options and mutually exclusive weight-loading policies."""
    runtime_allowed = {
        "seed",
        "experiment_name",
        "run_dir",
        "trainer",
        "checkpoint",
        "logging",
        "callbacks",
        "resume",
        "validate_before_fit",
        "test_weights",
        "save_rng",
        "evaluation_trainer",
        "inference",
        "final_validation",
    }
    if runtime.keys() - runtime_allowed:
        raise ValueError(
            f"Unknown runtime options: {sorted(runtime.keys() - runtime_allowed)}"
        )
    required = ("trainer", "checkpoint", "logging")
    if "experiment_name" not in runtime:
        required = ("run_dir", *required)
    for name in required:
        if name not in runtime:
            raise ValueError(f"runtime.{name} is required")
    for name in ("trainer", "checkpoint", "logging", "evaluation_trainer", "inference"):
        if name in runtime:
            _require_mapping(runtime[name], f"runtime.{name}")
    if runtime.get("inference", {}).keys() - {"precision", "matmul_tf32", "cudnn_tf32"}:
        raise ValueError(
            "runtime.inference accepts precision, matmul_tf32 and cudnn_tf32"
        )
    for location, specification in (
        ("runtime.evaluation_trainer", runtime.get("evaluation_trainer", {})),
        ("runtime.inference", runtime.get("inference", {})),
    ):
        precision = specification.get("precision")
        if precision not in (None, 32, "32", "32-true"):
            raise ValueError(f"{location}.precision must be 32-true for evaluation")
        if location == "runtime.inference" and any(
            specification.get(name) is True for name in ("matmul_tf32", "cudnn_tf32")
        ):
            raise ValueError("runtime.inference must disable TF32 for evaluation")
    if runtime.get("resume") and initialization.get("checkpoint"):
        raise ValueError(
            "Choose initialization.checkpoint for a new run or runtime.resume"
        )
    if runtime.get("final_validation", "scheduled") not in {
        "scheduled",
        "require",
        "force",
    }:
        raise ValueError("runtime.final_validation must be scheduled, require or force")
    for index, callback in enumerate(runtime.get("callbacks", [])):
        _validate_component(callback, f"runtime.callbacks[{index}]")
    logging_options = runtime["logging"]
    allowed_logging = {
        "logger",
        "tracking",
        "summary",
        "progress",
        "theme",
        "text_level",
        "data_facts",
    }
    if logging_options.keys() - allowed_logging:
        raise ValueError(
            f"Unknown runtime.logging options: {sorted(logging_options.keys() - allowed_logging)}"
        )
    logger = logging_options.get("logger")
    if isinstance(logger, Mapping):
        _validate_component(logger, "runtime.logging.logger")
    elif logger is not None and logger is not False:
        from pytorch_lightning.loggers.logger import Logger

        if not isinstance(logger, Logger):
            raise ValueError(
                "runtime.logging.logger must be a component mapping, Logger, false or null"
            )
    from .models.weights import validate_initialization

    validate_initialization(initialization)
    from pytorch_lightning import Trainer

    try:
        inspect.signature(Trainer).bind_partial(**runtime["trainer"])
    except TypeError as error:
        raise ValueError(f"Invalid runtime.trainer: {error}") from error


def _validate_model_and_loss(config: Mapping) -> None:
    """Validate the configured encoder and its primary/auxiliary objectives."""
    _validate_component(config["model"], "model", reserved=("freeze",))
    if config.get("loss") is not None:
        _validate_loss(config["loss"], "loss")


def _validate_loss(specification: Mapping, location: str) -> None:
    """Validate one independently configured metric objective composition."""
    loss = dict(_require_mapping(specification, location))
    _validate_component(loss, location)
    if isinstance(loss.get("primary_objective"), Mapping):
        _validate_component(loss["primary_objective"], f"{location}.primary_objective")
    for group in ("additional_losses", "auxiliaries"):
        for name, specification in loss.get(group, {}).items():
            objective_location = f"{location}.{group}.{name}"
            specification = _require_mapping(specification, objective_location)
            component_spec = (
                "type" in specification or {"module", "class"} <= specification.keys()
            )
            if "objective" in specification and not component_spec:
                unknown = specification.keys() - {"weight", "objective"}
                if unknown:
                    raise ValueError(
                        f"Unknown {objective_location} options: {sorted(unknown)}"
                    )
                _validate_component(
                    specification["objective"], f"{objective_location}.objective"
                )
            else:
                _validate_component(
                    specification, objective_location, reserved=("weight",)
                )


def _validate_optimization(config: Mapping) -> None:
    """Validate optimizer injection and scheduler units without building state."""
    optimizer = dict(config["optimizer"])
    optimizer.pop("groups", None)
    _validate_component(optimizer, "optimizer", injected=("params",))
    if config.get("scheduler") is not None:
        validate_scheduler(config["scheduler"])


def evaluation_sources(spec: Mapping[str, Any]) -> dict[str | None, Mapping[str, Any]]:
    """Return declared sources in loader order; ``None`` denotes a legacy split.

    Named sources cannot mix split fields at their parent. Names are single
    metric-path components, so independent sources never share a namespace.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("Evaluation split must be a mapping")
    if "sources" not in spec:
        return {None: spec}
    if set(spec) != {"sources"}:
        raise ValueError("sources cannot be mixed with split-level options")
    sources = spec["sources"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("sources must be a nonempty ordered mapping")
    for name, source in sources.items():
        if not isinstance(name, str) or not name or name.strip() != name or "/" in name:
            raise ValueError(f"Invalid evaluation source name: {name!r}")
        if not isinstance(source, Mapping):
            raise ValueError(f"Evaluation source {name!r} must be a mapping")
    return dict(sources)


def training_source_specs(
    spec: Mapping[str, Any],
) -> dict[str | None, Mapping[str, Any]]:
    """Return named training sources or one legacy source."""
    _require_mapping(spec, "data.train")
    if "sources" not in spec:
        return {None: spec}
    if set(spec) != {"sources"}:
        raise ValueError("Training sources cannot mix split-level options")
    sources = spec["sources"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("Training sources must be a nonempty mapping")
    for name, source in sources.items():
        if not isinstance(name, str) or not name or name.strip() != name or "/" in name:
            raise ValueError(f"Invalid training source name: {name!r}")
        if not isinstance(source, Mapping):
            raise ValueError(f"Training source {name!r} must be a mapping")
    return dict(sources)


def scheduler_definition(specification: Mapping) -> tuple[Callable, dict, dict]:
    """Return the constructor, its arguments and Lightning scheduling controls."""
    arguments = dict(specification)
    controls = {
        "interval": arguments.pop("interval", "epoch"),
        "frequency": arguments.pop("frequency", 1),
    }
    monitor = arguments.pop("monitor", None)
    if monitor is not None:
        controls["monitor"] = monitor
    if controls["interval"] not in {"step", "epoch"}:
        raise ValueError("scheduler.interval must be step or epoch")
    frequency = controls["frequency"]
    if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 1:
        raise ValueError("scheduler.frequency must be a positive integer")

    constructor, arguments = component_definition(arguments)
    if "total_steps" in inspect.signature(constructor).parameters:
        if controls["interval"] != "step" or frequency != 1:
            raise ValueError(
                "A total_steps scheduler requires interval=step, frequency=1"
            )
    return constructor, arguments, controls


def validate_scheduler(specification: Mapping) -> None:
    """Check factory arguments without constructing an optimizer or scheduler."""
    constructor, arguments, _ = scheduler_definition(specification)
    if "optimizer" in arguments:
        raise ValueError("The runtime supplies scheduler.optimizer")
    parameters = inspect.signature(constructor).parameters
    arguments = {**arguments, "optimizer": None}
    if "total_steps" in parameters:
        arguments.setdefault("total_steps", 1)
    try:
        inspect.signature(constructor).bind(**arguments)
    except TypeError as error:
        raise ValueError(f"Invalid scheduler: {error}") from error
