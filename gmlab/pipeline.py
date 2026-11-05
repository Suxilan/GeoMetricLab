"""Metric-learning routes shared by training, retrieval and feature extraction.

Pipelines own input/output contracts, not model parameters, GT interpretation or
the Lightning loop. Subclasses may adapt a new batch layout without modifying
model, loss, cache or Trainer implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import partial

import torch

from .components import build_component
from .training import adapters


def _required(config, paths, owner):
    missing = []
    for path in paths:
        value = config
        for part in path.split("."):
            value = value.get(part) if isinstance(value, Mapping) else None
        if value is None:
            missing.append(path)
    if missing:
        raise ValueError(f"{owner} requires: {', '.join(missing)}")


class Pipeline(ABC):
    """Configuration and assembly contract for a metric-learning experiment.

    A task pipeline binds model inputs/outputs. A joint pipeline composes such a
    task with source scheduling. Both own their required configuration fields.
    """

    required_fields = ()

    @abstractmethod
    def route(self, overrides=None):
        """Return the executable GeoPipeline task route for one source."""

    @abstractmethod
    def training_sources(self, config):
        """Validate and return this experiment's ordered source specifications."""

    @abstractmethod
    def assemble_train_loader(self, loaders, *, epoch_getter):
        """Assemble this experiment's native loader or finite source schedule."""

    def validate_config(self, config):
        """Validate this task's required fields before constructing any components."""
        _required(config, self.required_fields, type(self).__name__)
        for name, source in self.training_sources(config).items():
            location = "data.train" if name is None else f"data.train.sources.{name}"
            if source.get("adapters"):
                raise ValueError(
                    f"{location}: Source adapters cannot be mixed with pipeline routing"
                )
            if source.get("loss", config.get("loss")) is None:
                raise ValueError(f"{location} requires a source loss or a global loss")
            try:
                route = self.route(source.get("pipeline"))
                route.validate_source(source, location=location)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{location}: {error}") from error
        from .config import evaluation_sources

        for split in ("valid", "test"):
            if split in config["data"]:
                for source in evaluation_sources(config["data"][split]).values():
                    resolve_pipeline_route(config, source).validate_inference()


class GeoPipeline(Pipeline):
    """Bind one Tensor-input metric model to supervision and descriptor outputs.

    supervision selects labels, relations or named independent GT. training_output
    selects one descriptor or a named output mapping consumed by BoundObjective.
    retrieval_output explicitly selects a named [N,D] output for evaluation/cache.
    Model execution stays in the model; all adapters preserve original row order.
    """

    input_layout = "tensor"
    required_fields = ("model", "optimizer", "data.train")

    def route(self, overrides=None):
        """Bind source-specific task options without changing the pipeline type."""
        if overrides is None:
            return self
        if not isinstance(overrides, Mapping) or "type" in overrides:
            raise ValueError(
                "Source pipeline overrides cannot replace the pipeline type"
            )
        if not overrides:
            return self
        return _construct_pipeline({**self.configuration(), **overrides})

    def training_sources(self, config):
        from .config import training_source_specs

        train = config["data"]["train"]
        sources = training_source_specs(train)
        if "sources" in train:
            raise ValueError("Multiple training sources require MultiDatasetPipeline")
        return sources

    def assemble_train_loader(self, loaders, *, epoch_getter):
        return loaders[None]

    def validate_source(self, source, *, location):
        _required(source, ("dataset", "loader"), location)
        if self.supervision == "relations":
            _required(source, ("loader.collate_fn",), location)

    def __init__(
        self,
        *,
        supervision: str = "labels",
        training_output: str = "descriptor",
        retrieval_output: str | None = None,
    ):
        if supervision not in {"labels", "relations", "named"}:
            raise ValueError("pipeline.supervision must be labels, relations or named")
        if training_output not in {"descriptor", "named"}:
            raise ValueError("pipeline.training_output must be descriptor or named")
        if retrieval_output is not None and (
            not isinstance(retrieval_output, str) or not retrieval_output.strip()
        ):
            raise ValueError("pipeline.retrieval_output must be a nonempty output name")
        self.supervision = supervision
        self.training_output = training_output
        self.retrieval_output = retrieval_output
        self._specification = {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "supervision": supervision,
            "training_output": training_output,
            "retrieval_output": retrieval_output,
        }

    def configuration(self):
        """Return constructor options for source-local routing without mutation.

        Recipe construction records all declared subclass options. Subclasses
        composed directly in Python should extend this method for new options.
        """
        return dict(self._specification)

    def validate_model(self, model, *, inference=False):
        """Reject declared model/pipeline mismatches before reading a batch.

        External modules may declare input_layout ('tensor'/'grouped') and
        output_layout ('descriptor'/'named'). Undeclared contracts are checked
        from actual outputs, never guessed from class names or Dataset names.
        """
        layout = getattr(model, "input_layout", None)
        if layout is not None and layout != self.input_layout:
            raise ValueError(
                f"{type(self).__name__} supplies {self.input_layout} inputs, "
                f"but {type(model).__name__} requires {layout} inputs"
            )
        output = getattr(model, "output_layout", None)
        if not inference and output is not None and output != self.training_output:
            raise ValueError(
                f"pipeline.training_output={self.training_output} conflicts "
                f"with model.output_layout={output}"
            )

    def validate_inference(self):
        if self.training_output == "named" and self.retrieval_output is None:
            raise ValueError(
                "Named outputs require pipeline.retrieval_output for inference"
            )

    def validate_dataset(self, dataset, model):
        """Check declared modality coverage without decoding samples."""
        names = getattr(dataset, "declared_modalities", None)
        accepted = getattr(model, "input_modalities", None)
        if (
            self.input_layout == "grouped"
            and names is not None
            and accepted is not None
        ):
            missing = set(names) - set(accepted)
            if missing:
                raise ValueError(
                    f"Dataset modalities have no model route: {sorted(missing)}"
                )

    def training_adapters(self, loader):
        """Bind the collator's declared format once, before Trainer.fit."""
        layout = getattr(loader.collate_fn, "batch_format", None)
        self.validate_training_layout(layout)
        return partial(self.prepare_training, layout=layout), self.select_training

    def validate_training_layout(self, layout):
        allowed = {
            "labels": {None, "stacked_id"},
            "relations": {"subgraphs"},
            "named": {None, "named"},
        }
        if layout not in allowed[self.supervision]:
            raise ValueError(
                f"{type(self).__name__}(supervision={self.supervision}) cannot "
                f"consume collate batch_format={layout!r}"
            )

    def prepare_training(self, batch, *, layout=None):
        """Decode supervision using existing adapters; never modify its semantics."""
        if self.supervision == "relations":
            return adapters.relation_batch(batch)
        if self.supervision == "named":
            return adapters.named_batch(batch)
        processor = (
            adapters.stacked_id_batch if layout == "stacked_id" else adapters.id_batch
        )
        return processor(batch)

    def prepare_inference(self, batch):
        """Return Tensor inputs and stable manifest indices, without GT."""
        return adapters.image_batch(batch)

    def select_training(self, output, batch_info):
        selector = (
            adapters.named_embeddings
            if self.training_output == "named"
            else adapters.descriptor_embeddings
        )
        return selector(output, batch_info)

    def select_inference(self, output, batch_info):
        """Select exactly one retrieval descriptor space for both live and cache use."""
        self.validate_inference()
        if self.retrieval_output is not None:
            if not isinstance(output, Mapping) or self.retrieval_output not in output:
                raise ValueError(
                    f"Missing pipeline retrieval output: {self.retrieval_output!r}"
                )
            return adapters.tensor_embeddings(output[self.retrieval_output], batch_info)
        return adapters.descriptor_embeddings(output, batch_info)

    def inference_identity(self):
        """Describe inference-affecting settings for cache identity.

        Subclasses adding inference settings must extend this mapping. Training
        supervision does not affect feature calculation and is deliberately absent.
        """
        return {
            "input_layout": self.input_layout,
            "retrieval_output": self.retrieval_output,
        }

    def select_streams(self, output, batch_info):
        """Export explicitly declared named streams without changing their rows."""
        from .models.outputs import select_feature_streams

        return select_feature_streams(output, batch_info)


class MultimodalPipeline(GeoPipeline):
    """Route explicit modalities to a grouped-input model.

    Labels use grouped image collates; named GT uses arbitrary Tensor groups.
    Relations reuse the existing valid-node decoder, then group those valid nodes
    by supplied modality metadata. No modality is inferred from a source name.
    """

    input_layout = "grouped"

    def __init__(self, *, modalities=None, **kwargs):
        super().__init__(**kwargs)
        if modalities is not None and (
            not isinstance(modalities, (list, tuple))
            or len(modalities) < 2
            or any(not isinstance(name, str) or not name.strip() for name in modalities)
            or len(set(modalities)) != len(modalities)
        ):
            raise ValueError(
                "MultimodalPipeline.modalities requires at least two unique names"
            )
        self.modalities = None if modalities is None else tuple(modalities)
        self._specification["modalities"] = modalities

    def validate_source(self, source, *, location):
        super().validate_source(source, location=location)
        if self.modalities is None:
            raise ValueError(
                f"{location}: MultimodalPipeline requires pipeline.modalities"
            )
        _required(source, ("loader.collate_fn",), location)

    def validate_model(self, model, *, inference=False):
        super().validate_model(model, inference=inference)
        accepted = getattr(model, "input_modalities", None)
        if self.modalities is not None and accepted is not None:
            missing = set(self.modalities) - set(accepted)
            if missing:
                raise ValueError(
                    f"pipeline.modalities have no model route: {sorted(missing)}"
                )

    def validate_dataset(self, dataset, model):
        super().validate_dataset(dataset, model)
        declared = getattr(dataset, "declared_modalities", None)
        if self.modalities is not None and declared is not None:
            unknown = set(declared) - set(self.modalities)
            if unknown:
                raise ValueError(
                    f"Dataset modalities are absent from pipeline.modalities: {sorted(unknown)}"
                )

    def validate_training_layout(self, layout):
        allowed = {
            "labels": {"modality_id"},
            "relations": {"subgraphs"},
            "named": {None, "grouped_named"},
        }
        if layout not in allowed[self.supervision]:
            raise ValueError(
                f"MultimodalPipeline(supervision={self.supervision}) requires "
                f"a modality-aware collate; received batch_format={layout!r}"
            )

    def prepare_training(self, batch, *, layout=None):
        if self.supervision == "labels":
            return adapters.modality_id_batch(batch)
        if self.supervision == "named":
            return adapters.grouped_named_batch(batch)
        inputs, targets, info = adapters.relation_batch(batch)
        modalities = info.get("modalities")
        if modalities is None or len(modalities) != len(inputs):
            raise ValueError(
                "Multimodal relation batches require explicit valid-node modalities"
            )
        positions = {}
        for row, modality in enumerate(modalities):
            if not isinstance(modality, str) or not modality:
                raise ValueError("Each valid node requires a nonempty modality name")
            positions.setdefault(modality, []).append(row)
        positions = {
            name: torch.tensor(rows, dtype=torch.long, device=inputs.device)
            for name, rows in positions.items()
        }
        groups = {name: inputs[rows] for name, rows in positions.items()}
        return {"inputs": groups, "positions": positions}, targets, info

    def prepare_inference(self, batch):
        if not isinstance(batch, Mapping):
            raise ValueError(
                "Multimodal inference requires grouped inputs and positions"
            )
        inputs = batch.get("inputs", batch.get("images"))
        prepared, info = adapters.grouped_inference_batch({**batch, "inputs": inputs})
        info.update(
            {
                key: batch[key]
                for key in ("sample_ids", "modalities", "positions")
                if key in batch
            }
        )
        return prepared, info


class MultiDatasetPipeline(Pipeline):
    """Compose a metric task route with independent datasets and finite mixing.

    task owns model/input/supervision contracts. mixing owns source scheduling;
    per-source loss weights remain objective parameters, never sampling ratios.
    No batch concatenation, implicit cycling or cross-dataset GT is introduced.
    """

    required_fields = ("model", "optimizer", "data.train.sources")

    def __init__(self, *, task, mixing):
        self.task = _construct_pipeline(task)
        if not isinstance(self.task, GeoPipeline):
            raise ValueError(
                "Joint pipelines compose task routes, not nested joint pipelines"
            )
        if not isinstance(mixing, Mapping) or mixing.get("policy") != "interleave":
            raise ValueError("pipeline.mixing.policy must be interleave")
        if mixing.keys() - {"policy", "steps_per_source"}:
            raise ValueError("Unknown pipeline.mixing fields")
        limits = mixing.get("steps_per_source")
        if limits is not None and (
            not isinstance(limits, Mapping)
            or not limits
            or any(type(value) is not int or value < 1 for value in limits.values())
        ):
            raise ValueError(
                "mixing.steps_per_source must map source names to positive batch counts"
            )
        self.steps_per_source = None if limits is None else dict(limits)

    def route(self, overrides=None):
        return self.task.route(overrides)

    def training_sources(self, config):
        from .config import training_source_specs

        sources = training_source_specs(config["data"]["train"])
        if None in sources or len(sources) < 2:
            raise ValueError(
                "MultiDatasetPipeline requires at least two named data.train.sources"
            )
        if self.steps_per_source is not None and set(self.steps_per_source) != set(
            sources
        ):
            raise ValueError(
                "mixing.steps_per_source must name every training source exactly"
            )
        return sources

    def assemble_train_loader(self, loaders, *, epoch_getter):
        from .training.sources import MultiSourceLoader

        loader = MultiSourceLoader(
            loaders, epoch_getter=epoch_getter, steps_per_source=self.steps_per_source
        )
        if any(count < 1 for count in loader.budgets.values()):
            raise ValueError("Every joint-training source requires at least one batch")
        return loader


def _construct_pipeline(specification):
    if not isinstance(specification, Mapping) or "type" not in specification:
        raise ValueError("Pipeline requires a type and its task-specific options")
    pipeline = build_component(specification)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline.type must construct a metric Pipeline")
    pipeline._specification = dict(specification)
    return pipeline


def build_pipeline(config):
    """Build the experiment owner; return None for explicit-adapter recipes.

    Use resolve_pipeline_route for executable train/inference task adapters.
    """
    specification = config.get("pipeline")
    if specification is None:
        return None
    if not isinstance(specification, Mapping):
        raise ValueError("pipeline requires a type and explicit options")
    if config.get("adapters"):
        raise ValueError(
            "Choose pipeline routing or explicit adapters; do not mix them"
        )
    return _construct_pipeline(specification)


def resolve_pipeline_route(config, source=None):
    """Resolve the task route shared by train, evaluation and cache consumers."""
    pipeline = build_pipeline(config)
    source = source or {}
    if pipeline is None:
        if "pipeline" in source:
            raise ValueError("A source pipeline override requires a root pipeline")
        return None
    if source.get("adapters"):
        raise ValueError("Source adapters cannot be mixed with pipeline routing")
    return pipeline.route(source.get("pipeline"))


def validate_pipeline_config(config):
    """Check routing configuration without constructing models or reading data."""
    from .config import evaluation_sources, training_source_specs

    owner = build_pipeline(config)
    if owner is not None:
        owner.validate_config(config)
        return
    for split, specification in config.get("data", {}).items():
        if split not in {"train", "valid", "test"}:
            continue
        sources = (
            training_source_specs(specification)
            if split == "train"
            else evaluation_sources(specification)
        )
        for name, source in sources.items():
            try:
                pipeline = resolve_pipeline_route(config, source)
                if pipeline is not None and split != "train":
                    pipeline.validate_inference()
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"data.{split}.{name or 'default'}.pipeline: {error}"
                ) from error
