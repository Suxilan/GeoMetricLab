"""Explicit recipe assembly shared by train/evaluate; no dataset-name routing."""

import hashlib
import importlib
import inspect
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset

from ..components import (
    ADAPTERS,
    build_component,
    component_definition,
    resolve_adapter,
)
from ..config import evaluation_sources, training_source_specs
from ..datasets.utils.loaders import build_loader
from ..datasets.utils.geo_dataset import GeoDataset
from ..evaluation import evaluate_retrieval
from ..pipeline import build_pipeline, resolve_pipeline_route
from .checkpoints import initialize_model
from .module import EvaluationSource, MetricModule, TrainingSource
from .sources import MultiSourceLoader


def source_seed(seed: int, name: str | None) -> int:
    """Derive a stable named-source stream independent of declaration order."""
    if name is None:
        return seed
    digest = hashlib.sha256(f"gmlab-source-v1:{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def symbol(path: str) -> Callable:
    """Import an explicitly named callable without scanning modules."""
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"Expected a callable import path: {path!r}")
    module, name = path.rsplit(".", 1)
    adapter = getattr(importlib.import_module(module), name)
    if not callable(adapter):
        raise TypeError(f"Imported symbol is not callable: {path}")
    return adapter


def source_config(
    config: Mapping[str, Any], split: str, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve source-local evaluation/adapters without mutating the recipe."""
    resolved = dict(config)
    resolved["data"] = {**config["data"], split: dict(spec)}
    resolved["evaluation"] = {
        **config.get("evaluation", {}),
        **spec.get("evaluation", {}),
    }
    resolved["adapters"] = {**config.get("adapters", {}), **spec.get("adapters", {})}
    if "pipeline" in config:
        route = resolve_pipeline_route(config, spec)
        resolved["pipeline"] = route.configuration()
    return resolved


def build_evaluation_loaders(
    config: Mapping[str, Any],
    split: str,
    *,
    seed: int | None = None,
    model: torch.nn.Module | None = None,
) -> tuple[
    dict[str | None, Dataset],
    DataLoader | list[DataLoader],
    EvaluationSource | dict[str, EvaluationSource],
]:
    """Assemble independent retrieval sources and native Lightning loaders.

    Return datasets keyed by declared name (``None`` for legacy), a single
    loader or ordered loader list, and the corresponding MetricModule contract.
    Each source owns its protocol and inference adapters; rows are never merged.
    """
    datasets, loaders, evaluations = {}, [], {}
    for name, spec in evaluation_sources(config["data"][split]).items():
        local = source_config(config, split, spec)
        loader_spec = dict(spec.get("loader", {}))
        if "sampler" not in loader_spec and "batch_sampler" not in loader_spec:
            loader_spec["sampler"] = {
                "type": "gmlab.datasets.utils.sampling.ExactDistributedSampler"
            }
        source_spec = {**spec, "loader": loader_spec}
        dataset, loader = build_loader(
            source_spec,
            config["data"].get("transforms"),
            seed=None if seed is None else source_seed(seed, name),
        )
        datasets[name] = dataset
        loaders.append(loader)
        size, evaluator = retrieval_evaluation(dataset, local["evaluation"])
        _, embedding_adapter = resolve_embedding_adapter(local, inference=True)
        pipeline = resolve_pipeline_route(local)
        if pipeline is not None:
            pipeline.validate_inference()
            if model is not None:
                pipeline.validate_model(model, inference=True)
                pipeline.validate_dataset(dataset, model)
        evaluations[name] = EvaluationSource(
            size,
            evaluator,
            batch_adapter=(
                pipeline.prepare_inference
                if pipeline is not None
                else resolve_adapter(local["adapters"]["inference_batch"])
                if "inference_batch" in local.get("adapters", {})
                else loader_batch_processor(loader, inference=True)
            ),
            embedding_adapter=embedding_adapter,
        )
    if None in datasets:
        return datasets, loaders[0], evaluations[None]
    return datasets, loaders, evaluations


def build_model(config: Mapping[str, Any]) -> torch.nn.Module:
    """Construct model structure; do not load experiment initialization weights."""
    specification = config["model"]
    if isinstance(specification, Mapping):
        specification = dict(specification)
        specification.pop("freeze", None)
    return build_component(specification)


def freeze_parameters(model: torch.nn.Module, prefixes: list[str]) -> None:
    """Disable gradients for each explicitly selected model parameter prefix."""
    for prefix in prefixes:
        matching_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name == prefix or name.startswith(prefix + ".")
        ]
        if not matching_parameters:
            raise ValueError(f"Frozen parameter prefix matches nothing: {prefix}")
        for parameter in matching_parameters:
            parameter.requires_grad_(False)


RetrievalEvaluation = tuple[int, Callable[[torch.Tensor], dict[str, float]]]


def retrieval_evaluation(
    dataset: Dataset, options: Mapping[str, Any]
) -> RetrievalEvaluation:
    """Return (manifest_size, evaluator) for MetricModule.evaluation[stage].

    The callable consumes aligned float embeddings [N,D], where N is the
    returned size, and returns macro retrieval percentages/counts. Dataset-level
    eval_meta owns query/gallery positions, GT and self policy. Search options
    cannot override those identities/annotations. No Dataset is read by the
    evaluator and no temporary embedding state is owned here.
    """
    if not hasattr(dataset, "eval_meta"):
        raise ValueError("Supervised retrieval requires dataset-level eval_meta/GT")
    overlap = set(dataset.eval_meta) & set(options)
    if overlap:
        raise ValueError(
            f"Protocol override conflicts with dataset metadata: {overlap}"
        )
    return len(dataset), partial(evaluate_retrieval, **dataset.eval_meta, **options)


def loader_batch_processor(loader, *, inference=False):
    """Select processing from the collator's declared output layout."""
    layout = getattr(loader.collate_fn, "batch_format", None)
    names = (
        {None: "image", "modality_id": "modality_image"}
        if inference
        else {
            None: "id",
            "subgraphs": "relation",
            "stacked_id": "stacked_id",
            "modality_id": "modality_id",
        }
    )
    if layout not in names:
        raise ValueError(f"Unsupported collator batch_format: {layout!r}")
    return resolve_adapter(names[layout])


class TrainingComponents(NamedTuple):
    """Named results of experiment assembly, usable directly from Python."""

    module: MetricModule
    train_loader: DataLoader | MultiSourceLoader
    valid_loader: DataLoader | list[DataLoader] | None
    test_loader: DataLoader | list[DataLoader] | None
    valid_metadata: dict[str, Any]


DEFAULT_EMBEDDING_ADAPTER = "gmlab.training.adapters.descriptor_embeddings"


def resolve_embedding_adapter(
    config: Mapping[str, Any], *, inference: bool = False
) -> tuple[str | dict, Callable]:
    """Return the declared (import_path, adapter) shared by all entrypoints.

    Training uses adapters.embedding. Inference uses adapters.inference_embedding
    when supplied, otherwise the training selection. The callable consumes
    (model_output, batch_info) and returns aligned embeddings; inference must
    select one Tensor [N,D]. No model-output or dataset-name guessing occurs.
    """
    pipeline = resolve_pipeline_route(config)
    if pipeline is not None:
        from ..extraction_identity import describe_adapter

        adapter = pipeline.select_inference if inference else pipeline.select_training
        return describe_adapter(adapter), adapter
    path = config.get("adapters", {}).get("embedding", DEFAULT_EMBEDDING_ADAPTER)
    if inference:
        path = config.get("adapters", {}).get("inference_embedding", path)
    path = ADAPTERS.get(path, path)
    if not isinstance(path, str):
        raise TypeError("embedding_adapter must be a callable import path")
    return path, symbol(path)


def bind_dataset_relations(criterion: torch.nn.Module, loader: DataLoader) -> None:
    """Bind read-only GT queries using the collator's declared decoding options.

    No source is inferred from objective names. Each query is selected later
    using the provenance carried by its corresponding supervision value.
    Custom collators without relation metadata retain their own supervision.
    """
    options = getattr(loader.collate_fn, "relation_options", None)
    factories = getattr(loader.dataset, "relation_resolvers", None)
    factory = getattr(loader.dataset, "relation_resolver", None)
    if options is None:
        return
    if callable(factories):
        resolvers = tuple(factories(**options))
    elif callable(factory):
        resolvers = (factory(**options),)
    else:
        return
    keys = [resolver.key for resolver in resolvers]
    if any(key is None for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Dataset relation queries must have distinct source keys")
    for objective in criterion.modules():
        bind = getattr(objective, "bind_relation_resolver", None)
        if callable(bind):
            for resolver in resolvers:
                bind(resolver)


def build_train_loader(spec, transforms, *, seed, multi_source):
    """Choose an epoch sampler only when Dataset multiplier or multi-source needs it."""
    dataset_spec = spec["dataset"]
    if isinstance(dataset_spec, Mapping):
        dataset_type, dataset_arguments = component_definition(dataset_spec)
    else:
        dataset_type = type(dataset_spec)
        dataset_arguments = {
            "sampling_multiplier": getattr(dataset_spec, "sampling_multiplier", 1.0)
        }
    options = dict(spec.get("loader", {}))
    multiplier = dataset_arguments.get("sampling_multiplier", 1.0)
    is_geo_dataset = inspect.isclass(dataset_type) and issubclass(
        dataset_type, GeoDataset
    )
    if is_geo_dataset and (multi_source or multiplier != 1.0):
        if "sampler" not in options and "batch_sampler" not in options:
            sampler = "GeoEpochSampler"
            shuffle = options.pop("shuffle", True)
            options["sampler"] = {"type": sampler, "seed": seed, "shuffle": shuffle}
    elif multi_source and "sampler" not in options and "batch_sampler" not in options:
        raise ValueError(
            "Multi-source readers outside GeoDataset require an explicit "
            "rank-aware sampler or batch_sampler"
        )
    dataset, loader = build_loader({**spec, "loader": options}, transforms, seed=seed)
    if isinstance(dataset, GeoDataset) and dataset.sampling_multiplier != 1.0:
        compatible = bool(
            getattr(loader.sampler, "supports_sampling_multiplier", False)
        )
        compatible |= bool(
            getattr(loader.batch_sampler, "supports_sampling_multiplier", False)
        )
        if not compatible:
            raise ValueError(
                "Non-unit Dataset sampling_multiplier requires an epoch sampler "
                "that consumes it"
            )
    return dataset, loader


def _build_training_source(config, specification, loader, model):
    criterion = build_component(specification.get("loss", config.get("loss")))
    bind_dataset_relations(criterion, loader)
    pipeline = resolve_pipeline_route(config, specification)
    if pipeline is not None:
        pipeline.validate_model(model)
        pipeline.validate_dataset(loader.dataset, model)
        batch_adapter, embedding_adapter = pipeline.training_adapters(loader)
        return TrainingSource(criterion, batch_adapter, embedding_adapter)
    adapters = {**config.get("adapters", {}), **specification.get("adapters", {})}
    batch_adapter = (
        resolve_adapter(adapters["batch"])
        if "batch" in adapters
        else loader_batch_processor(loader)
    )
    return TrainingSource(
        criterion, batch_adapter, resolve_embedding_adapter({"adapters": adapters})[1]
    )


def _training_bindings(config, specifications, loaders, model):
    """Normalize sources once; retain the original single-source checkpoint keys."""
    sources = {}
    for name, spec in specifications.items():
        try:
            sources[name] = _build_training_source(config, spec, loaders[name], model)
        except (TypeError, ValueError) as error:
            if "pipeline" not in config:
                raise
            raise ValueError(f"data.train.{name or 'default'}: {error}") from error
    if None in sources:
        source, dataset = sources[None], loaders[None].dataset
        return {
            "criterion": source.criterion,
            "batch_adapter": source.batch_adapter,
            "embedding_adapter": source.embedding_adapter,
            "class_geo_ids": getattr(dataset, "class_geo_ids", None),
            "class_vocabularies": getattr(dataset, "class_vocabularies", None),
        }
    return {
        "training_sources": sources,
        "class_vocabularies": {
            name: tuple(loader.dataset.class_geo_ids)
            for name, loader in loaders.items()
            if getattr(loader.dataset, "class_geo_ids", None)
        },
    }


def build_training_components(
    config: Mapping[str, Any],
) -> TrainingComponents:
    """Assemble the model, objective and explicitly requested data loaders.

    Only train supplies supervision/criterion. data.valid selects the loader used
    for checkpoint selection; dataset metadata preserves its source split name.
    A source named "test" may explicitly serve that role (e.g. GL3D's held-out
    scenes). This does not make later evaluation on it an independent test.
    Optional post-fit test requires an explicit best/last selection.
    The named result carries static validation provenance separately from
    per-batch targets. Callers own these loaders; PL owns runtime/state.
    """
    data = config["data"]
    transforms = data.get("transforms")
    runtime = config.get("runtime", {})
    seed = runtime.get("seed", 81)
    owner = build_pipeline(config)
    if owner is not None:
        owner.validate_config(config)
    source_specs = (
        owner.training_sources(config)
        if owner is not None
        else training_source_specs(data["train"])
    )
    pipeline = resolve_pipeline_route(config)
    model = build_model(config) if pipeline is not None else None
    if pipeline is not None:
        pipeline.validate_model(model)
    multi_source = None not in source_specs
    train_loaders = {}
    for name, source in source_specs.items():
        _, train_loaders[name] = build_train_loader(
            source, transforms, seed=source_seed(seed, name), multi_source=multi_source
        )
    train_loader = train_loaders[None] if not multi_source else None
    val_loader = None
    valid_metadata = {}
    evaluations = {}
    if "valid" in data:
        datasets, val_loader, evaluations["val"] = build_evaluation_loaders(
            config, "valid", seed=seed + 1, model=model
        )
        valid_metadata = {
            name: getattr(dataset, "metadata", {}) for name, dataset in datasets.items()
        }
        if None in valid_metadata:
            valid_metadata = valid_metadata[None]
    test_loader = None
    if "test" in data:
        _, test_loader, evaluations["test"] = build_evaluation_loaders(
            config, "test", seed=seed + 2, model=model
        )
        if not runtime.get("trainer", {}).get("fast_dev_run") and runtime.get(
            "test_weights"
        ) not in {"best", "last"}:
            raise ValueError("Post-fit test requires explicit best/last selection")
    _, inference_embedding_adapter = resolve_embedding_adapter(config, inference=True)
    if model is None:
        model = build_model(config)
    freeze_parameters(model, config["model"].get("freeze", []))
    initialize_model(model, config.get("initialization"))
    adapter_options = config.get("adapters", {})
    inference_batch_adapter = (
        pipeline.prepare_inference
        if pipeline is not None
        else resolve_adapter(adapter_options.get("inference_batch", "image"))
    )
    module = MetricModule(
        model=model,
        **_training_bindings(config, source_specs, train_loaders, model),
        optimizer=config["optimizer"],
        scheduler=config.get("scheduler"),
        inference_batch_adapter=inference_batch_adapter,
        inference_embedding_adapter=inference_embedding_adapter,
        evaluation=evaluations,
        inference_options=config.get("runtime", {}).get("inference", {}),
    )
    module.pipeline = owner
    module.pipeline_configuration = (
        {
            "default": dict(config["pipeline"]),
            "sources": {
                name or "default": dict(spec.get("pipeline", {}))
                for name, spec in source_specs.items()
            },
        }
        if pipeline is not None
        else None
    )
    if owner is not None:
        train_loader = owner.assemble_train_loader(
            train_loaders, epoch_getter=lambda: module.current_epoch
        )
    elif multi_source:
        train_loader = MultiSourceLoader(
            train_loaders, epoch_getter=lambda: module.current_epoch
        )
    return TrainingComponents(
        module=module,
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        valid_metadata=valid_metadata,
    )
