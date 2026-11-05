"""Lightning integration for injected metric supervision and objectives."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from pytorch_metric_learning.utils.distributed import DistributedLossWrapper
from torch import nn

from ..models.identity import model_identity_record
from ..utils.distributed import check_rank_consistency, gather_context
from ..utils.precision import (
    resolve_inference_precision,
    inference_precision_context,
    inference_precision_record,
)

from .adapters import id_batch, image_batch, scalar_reports, tensor_embeddings
from .collection import evaluate_records
from .optimization import OptimizerSpec, SchedulerSpec, configure_optimization


@dataclass(frozen=True)
class TrainingSource:
    """Bind one criterion to batch decoding and differentiable output selection.

    batch_adapter returns (model input, criterion kwargs, batch information).
    embedding_adapter consumes (model output, batch information). Criteria are
    registered by MetricModule, not by this immutable routing record.
    """

    criterion: nn.Module
    batch_adapter: Callable = id_batch
    embedding_adapter: Callable = tensor_embeddings

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, nn.Module):
            raise TypeError("Training source criterion must be an nn.Module")
        if not callable(self.batch_adapter) or not callable(self.embedding_adapter):
            raise TypeError("Training source adapters must be callable")


@dataclass(frozen=True)
class EvaluationSource:
    """One retrieval source with its own [0,size) row space and optional adapters.

    evaluator consumes aligned descriptors [size,D] and returns scalar metrics.
    Omitted adapters inherit the module's inference adapters. Source names belong
    to the containing ordered mapping, independently of local sample identities.
    """

    size: int
    evaluator: Callable
    batch_adapter: Callable | None = None
    embedding_adapter: Callable | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise ValueError("Evaluation source size must be a nonnegative integer")
        if not callable(self.evaluator):
            raise TypeError("Evaluation source evaluator must be callable")
        for adapter in (self.batch_adapter, self.embedding_adapter):
            if adapter is not None and not callable(adapter):
                raise TypeError("Evaluation source adapters must be callable")


class _EvaluationCleanup(pl.Callback):
    """Discard partial retrieval state when Lightning aborts any entrypoint."""

    def on_exception(self, trainer, pl_module, exception) -> None:
        for stage in pl_module.evaluation:
            pl_module._clear_evaluation(stage)


class MetricModule(pl.LightningModule):
    """Register model/criterion; let Lightning own backward and optimization.

    batch_adapter returns (model input, criterion keyword arguments, batch info).
    embedding_adapter selects training outputs; inference_embedding_adapter
    selects a Tensor [N,D] for retrieval (defaults to the training adapter). Criterion returns
    a scalar Tensor or a dictionary of scalar Tensors containing ``loss``.
    GeoMetricLoss owns weighting and distributed backward/reporting semantics.
    The criterion call binds Lightning's public tensor gather for components
    that prepare distributed candidates. PML wrappers retain their own gather;
    ordinary criteria need no runtime keyword or new base class.

    Reports are valid-image-weighted batch means. Inference consumes its own
    adapter and stable indices, without criterion or optimizer construction.
    Registered model/criterion parameters use PyTorch's shared deduplication.
    evaluation maps val/test to (size, evaluator) for a legacy single source,
    or to an ordered mapping of source names to EvaluationSource records.
    Such stages use the inference adapter, never training supervision/criterion.
    Criterion and optimizer are optional for independent test/predict. Sanity
    checks only exercise input/model wiring, since PL may take partial batches.
    """

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist class semantics alongside model and criterion parameters."""
        if getattr(self, "pipeline_configuration", None) is not None:
            checkpoint["gmlab_pipeline"] = self.pipeline_configuration
        if self.class_geo_ids is not None:
            checkpoint["gmlab_class_geo_ids"] = list(self.class_geo_ids)
        if self.class_vocabularies:
            checkpoint["gmlab_class_vocabularies"] = self.class_vocabularies
        if self.named_training:
            checkpoint["gmlab_training_sources"] = list(self.training_sources)
            checkpoint["gmlab_source_counts"] = self._source_counts
            checkpoint["gmlab_source_count_epoch"] = self.current_epoch

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Reject training resume with missing or changed class semantics.

        Independent inference has no declared training vocabulary and does not
        consume criterion/proxy semantics. Model-only initialization is separate
        from Lightning's full-state training resume.
        """
        if self.training_sources and "gmlab_pipeline" in checkpoint:
            if checkpoint["gmlab_pipeline"] != getattr(
                self, "pipeline_configuration", None
            ):
                raise ValueError(
                    "Checkpoint pipeline routing differs from the training recipe"
                )
        if self.class_vocabularies:
            saved_vocabularies = checkpoint.get("gmlab_class_vocabularies")
            if saved_vocabularies != self.class_vocabularies:
                raise ValueError(
                    "Checkpoint named class vocabularies differ or are missing"
                )
        if self.named_training and (
            checkpoint.get("gmlab_training_sources") != list(self.training_sources)
        ):
            raise ValueError("Checkpoint training sources differ")
        if self.named_training:
            self._source_counts = checkpoint.get(
                "gmlab_source_counts", dict.fromkeys(self.training_sources, 0)
            )
            self._source_count_epoch = checkpoint.get("gmlab_source_count_epoch")
        if self.class_geo_ids is None:
            return
        saved = checkpoint.get("gmlab_class_geo_ids")
        if not isinstance(saved, (list, tuple)) or tuple(saved) != self.class_geo_ids:
            raise ValueError(
                "Checkpoint class vocabulary is missing or differs from the dataset; "
                "full training resume requires identical ordered class GeoIDs"
            )

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module | None = None,
        optimizer: OptimizerSpec | None = None,
        *,
        scheduler: SchedulerSpec | None = None,
        batch_adapter: Callable = id_batch,
        embedding_adapter: Callable = tensor_embeddings,
        inference_batch_adapter: Callable = image_batch,
        inference_embedding_adapter: Callable | None = None,
        evaluation: Mapping | None = None,
        inference_options: Mapping | None = None,
        class_geo_ids: tuple[str, ...] | None = None,
        class_vocabularies: Mapping[str, tuple[str, ...]] | None = None,
        training_sources: Mapping[str, TrainingSource] | None = None,
    ) -> None:
        super().__init__()
        for name, adapter in (
            ("batch_adapter", batch_adapter),
            ("embedding_adapter", embedding_adapter),
        ):
            if not callable(adapter):
                raise TypeError(f"{name} must be callable")
        self._configure_vocabularies(class_geo_ids, class_vocabularies)
        self.model = model
        self.criterion = criterion
        self.named_training = training_sources is not None
        if training_sources is not None and criterion is not None:
            raise ValueError("Supply training_sources or criterion, not both")
        self.training_sources = dict(training_sources or {})
        if not self.named_training and criterion is not None:
            self.training_sources["default"] = TrainingSource(
                criterion, batch_adapter, embedding_adapter
            )
        if any(
            not isinstance(source, TrainingSource)
            for source in self.training_sources.values()
        ):
            raise TypeError("training_sources values must be TrainingSource records")
        self.source_criteria = nn.ModuleDict(
            {name: source.criterion for name, source in self.training_sources.items()}
            if self.named_training
            else {}
        )
        self._source_counts = dict.fromkeys(self.training_sources, 0)
        self._source_count_epoch = None
        self.current_step_metrics = {}
        if not isinstance(self.model, nn.Module) or (
            self.criterion is not None and not isinstance(self.criterion, nn.Module)
        ):
            raise TypeError(
                "model and any supplied criterion must be nn.Module instances"
            )
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.inference_batch_adapter = inference_batch_adapter
        self.inference_embedding_adapter = (
            embedding_adapter
            if inference_embedding_adapter is None
            else inference_embedding_adapter
        )
        for name in (
            "inference_batch_adapter",
            "inference_embedding_adapter",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        self.inference_options = dict(inference_options or {})
        self.inference_execution = None
        self._source_execution = {"val": {}, "test": {}}
        self._configure_evaluation(evaluation)

    def _configure_evaluation(self, evaluation):
        """Normalize independent retrieval sources once at construction."""
        self.evaluation = dict(evaluation or {})
        self._evaluation_sources = {}
        self._named_evaluation = set()
        for stage, specification in self.evaluation.items():
            if stage not in {"val", "test"}:
                raise ValueError("Evaluation stages must be val or test")
            if isinstance(specification, Mapping):
                if not specification:
                    raise ValueError("Named evaluation sources cannot be empty")
                sources = dict(specification)
                for name, source in sources.items():
                    if (
                        not isinstance(name, str)
                        or not name
                        or name != name.strip()
                        or "/" in name
                    ):
                        raise ValueError(
                            "Evaluation source names must be nonempty without '/'"
                        )
                    if not isinstance(source, EvaluationSource):
                        raise TypeError(
                            "Named evaluation requires EvaluationSource values"
                        )
                self._named_evaluation.add(stage)
            else:
                source = (
                    specification
                    if isinstance(specification, EvaluationSource)
                    else EvaluationSource(*specification)
                )
                sources = {None: source}
            self._evaluation_sources[stage] = sources
        self.evaluation_results = {}
        self._records = {"val": [], "test": []}

    def _configure_vocabularies(self, class_geo_ids, class_vocabularies):
        """Validate ordered class identities independently of routing."""
        if class_geo_ids is not None:
            if (
                not isinstance(class_geo_ids, tuple)
                or not class_geo_ids
                or any(not isinstance(key, str) or not key for key in class_geo_ids)
                or len(set(class_geo_ids)) != len(class_geo_ids)
            ):
                raise ValueError("class_geo_ids must be unique nonempty strings")
        self.class_vocabularies = {}
        for name, vocabulary in (class_vocabularies or {}).items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(vocabulary, tuple)
                or not vocabulary
                or any(not isinstance(key, str) or not key for key in vocabulary)
                or len(set(vocabulary)) != len(vocabulary)
            ):
                raise ValueError(
                    "Named class vocabularies require unique nonempty GeoIDs"
                )
            self.class_vocabularies[name] = list(vocabulary)
        self.class_geo_ids = class_geo_ids

    @property
    def batch_adapter(self):
        """Expose the single-source Python convenience adapter."""
        return self.training_sources["default"].batch_adapter

    @property
    def embedding_adapter(self):
        """Expose the single-source output selection or inference selection."""
        source = self.training_sources.get("default")
        return source.embedding_adapter if source else self.inference_embedding_adapter

    def configure_callbacks(self) -> list[pl.Callback]:
        return [_EvaluationCleanup()] if self.evaluation else []

    def evaluation_source_name(self, stage: str, dataloader_idx: int) -> str | None:
        """Resolve Lightning's loader position to a declared retrieval source."""
        sources = self._evaluation_sources.get(stage)
        count = len(sources) if sources is not None else 1
        if (
            isinstance(dataloader_idx, bool)
            or not isinstance(dataloader_idx, int)
            or not 0 <= dataloader_idx < count
        ):
            raise ValueError(f"Invalid {stage} dataloader_idx: {dataloader_idx!r}")
        if sources is None:
            return None
        return tuple(sources)[dataloader_idx]

    def forward(self, inputs: Any) -> Any:
        return self.model(inputs)

    def on_fit_start(self) -> None:
        criteria = tuple(source.criterion for source in self.training_sources.values())
        for criterion in criteria:
            if (
                self.trainer.world_size > 1
                and isinstance(criterion, DistributedLossWrapper)
                and any(p.requires_grad for p in criterion.parameters())
            ):
                raise ValueError(
                    "Native PML distributed wrapping of trainable objectives has "
                    "not been validated here; inject a local objective or a verified adapter"
                )
        if self.named_training and self.trainer.world_size > 1:
            distributed_model = self.trainer.strategy.model
            if not isinstance(distributed_model, nn.parallel.DistributedDataParallel):
                raise ValueError("Multi-source training requires a DDP strategy")
            if not distributed_model.find_unused_parameters:
                raise ValueError(
                    "Multi-source DDP requires find_unused_parameters=True for "
                    "source-specific objectives and model branches"
                )

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        return configure_optimization(
            self,
            self.optimizer_factory,
            self.scheduler_factory,
            self._optimizer_step_horizon,
        )

    def _optimizer_step_horizon(self) -> int:
        return self.trainer.estimated_stepping_batches

    def _step(self, batch: Any, stage: str) -> tuple[torch.Tensor, Any]:
        source_name = batch["source"] if self.named_training else "default"
        if source_name not in self.training_sources:
            raise ValueError(f"Unknown training source: {source_name!r}")
        source = self.training_sources[source_name]
        if self.named_training:
            batch = batch["batch"]
        criterion = source.criterion
        model_inputs, loss_kwargs, batch_info = source.batch_adapter(batch)
        model_output = self(model_inputs)
        embeddings = source.embedding_adapter(model_output, batch_info)
        with gather_context(self.all_gather):
            loss_output = criterion(embeddings, **loss_kwargs)

        reports = scalar_reports(loss_output, criterion, self.trainer.world_size)
        loss = loss_output if torch.is_tensor(loss_output) else loss_output["loss"]
        self.current_step_metrics = {}
        for name, value in reports.items():
            metric_name = (
                f"{stage}/{source_name}/{name}"
                if self.named_training
                else f"{stage}/{name}"
            )
            if self.named_training:
                # Lightning retains each metric's last step value across batches.
                # Publish sparse source steps directly; keep epoch aggregation native.
                reduced = self.trainer.strategy.reduce(value.detach(), reduce_op="mean")
                self.current_step_metrics[f"{metric_name}_step"] = reduced.item()
            self.log(
                f"{metric_name}_epoch" if self.named_training else metric_name,
                value,
                on_step=stage == "train" and not self.named_training,
                on_epoch=True,
                batch_size=batch_info["num_images"],
                sync_dist=True,
                prog_bar=name in {"loss", "miner/rate"},
                logger=bool(self.trainer.loggers),
            )
        return loss, embeddings

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Emit only the active source's detached step values on global rank zero."""
        if self.named_training and self.trainer.is_global_zero:
            for logger in self.trainer.loggers:
                logger.log_metrics(self.current_step_metrics, step=self.global_step)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict:
        return self._predict_batch(batch)

    def _predict_batch(
        self,
        batch: Any,
        source: EvaluationSource | None = None,
        *,
        full_precision: bool = False,
    ) -> dict:
        batch_adapter = (
            source.batch_adapter
            if source is not None and source.batch_adapter is not None
            else self.inference_batch_adapter
        )
        embedding_adapter = (
            source.embedding_adapter
            if source is not None and source.embedding_adapter is not None
            else self.inference_embedding_adapter
        )
        model_inputs, batch_info = batch_adapter(batch)
        options = dict(self.inference_options)
        native_precision = (
            self.trainer.precision if self._trainer is not None else "32-true"
        )
        requested = options.pop("precision", None)
        if full_precision:
            if requested not in (None, 32, "32", "32-true"):
                raise ValueError("Retrieval evaluation requires full precision")
            if options.get("matmul_tf32") or options.get("cudnn_tf32"):
                raise ValueError("Retrieval evaluation requires TF32 to be disabled")
            options["matmul_tf32"] = False
            options["cudnn_tf32"] = False
            inference_precision = "32-true"
        else:
            inference_precision = native_precision if requested is None else requested
        policy = resolve_inference_precision(
            self.device, inference_precision, **options
        )
        if (
            not full_precision
            and self._trainer is not None
            and policy["precision"] != native_precision
        ):
            raise ValueError("Inference precision conflicts with Trainer precision")
        with inference_precision_context(self.device, policy):
            self.inference_execution = inference_precision_record(self.device, policy)
            if full_precision:
                model_inputs = apply_to_collection(
                    model_inputs,
                    torch.Tensor,
                    lambda value: value.float() if value.is_floating_point() else value,
                )
            model_output = self(model_inputs)
            embeddings = embedding_adapter(model_output, batch_info)
        if not torch.is_tensor(embeddings) or embeddings.ndim != 2:
            raise ValueError(
                "Inference adapter must select one Tensor [N,D] retrieval head"
            )
        self.inference_execution["output_dtype"] = str(embeddings.dtype)
        return {"embeddings": embeddings.detach(), "index": batch_info["index"]}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Execute one validated source and count its completed forward step."""
        loss = self._step(batch, "train")[0]
        name = batch["source"] if self.named_training else "default"
        self._source_counts[name] += 1
        return loss

    def on_train_epoch_start(self) -> None:
        """Reset source counts at a new epoch, retaining resumed partial counts."""
        if self._source_count_epoch != self.current_epoch:
            self._source_counts = dict.fromkeys(self.training_sources, 0)
            self._source_count_epoch = self.current_epoch

    def on_train_epoch_end(self) -> None:
        """Report actual batch consumption independently of configured exposure."""
        if not self.named_training:
            return
        total = sum(self._source_counts.values())
        for name, count in self._source_counts.items():
            self.log(
                f"train/{name}/batches",
                float(count),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"train/{name}/share",
                count / total if total else 0.0,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def _clear_evaluation(self, stage: str) -> None:
        self._records[stage] = []
        self.evaluation_results.pop(stage, None)
        self._source_execution[stage] = {}

    def _start_evaluation(self, stage: str) -> None:
        self._clear_evaluation(stage)
        sources = self._evaluation_sources.get(stage)
        # Validate the complete roster before any per-source payload collective.
        error = None
        if sources is not None:
            loader_batches = (
                self.trainer.num_val_batches
                if stage == "val"
                else self.trainer.num_test_batches
            )
            if len(loader_batches) != len(sources):
                error = "ValueError: Evaluation loader count differs from source count"
        check_rank_consistency(
            error=error,
            signature=(
                stage,
                tuple((name, source.size) for name, source in sources.items()),
            )
            if sources is not None
            else (stage, None),
        )
        if sources is not None:
            self._records[stage] = {name: [] for name in sources}

    def _evaluation_step(
        self, batch: Any, batch_idx: int, stage: str, dataloader_idx: int
    ) -> dict:
        sources = self._evaluation_sources[stage]
        try:
            name = self.evaluation_source_name(stage, dataloader_idx)
            output = self._predict_batch(batch, sources[name], full_precision=True)
            self._source_execution[stage][name] = dict(self.inference_execution)
            self._records[stage][name].append(
                (output["index"].detach().cpu(), output["embeddings"].cpu())
            )
            return output
        except Exception:
            self._clear_evaluation(stage)
            raise

    def _finish_evaluation(self, stage: str) -> None:
        sources = self._evaluation_sources.get(stage)
        if sources is None:
            return
        if self.trainer.sanity_checking or self.trainer.fast_dev_run:
            self._clear_evaluation(stage)
            return
        records = self._records[stage]
        execution = self._source_execution[stage]
        self._clear_evaluation(stage)
        source_results = {}
        metrics = {}
        # Complete all sources before publishing anything: a failed later source
        # must not leave a partial evaluation eligible for checkpoint selection.
        for name, source in sources.items():
            values = evaluate_records(records[name], source.size, source.evaluator)
            source_results[name] = {
                "manifest_size": source.size,
                "metrics": values,
                "inference_execution": execution.get(name),
            }
            prefix = f"{name}/" if stage in self._named_evaluation else ""
            metrics.update({f"{prefix}{key}": value for key, value in values.items()})
        identity = None
        error = None
        try:
            identity = model_identity_record(self.model)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        check_rank_consistency(error=error)
        for name, value in metrics.items():
            self.log(
                f"{stage}/{name}",
                torch.tensor(value, dtype=torch.float64, device=self.device),
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
                logger=bool(self.trainer.loggers),
            )
        result = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "weights_sha256": identity["sha256"],
            "model_identity": identity,
            "metrics": metrics,
        }
        if stage in self._named_evaluation:
            for source_result in source_results.values():
                source_result["weights_sha256"] = identity["sha256"]
                source_result["model_identity"] = identity
            result["sources"] = source_results
        self.evaluation_results[stage] = result

    def on_validation_epoch_start(self) -> None:
        self._start_evaluation("val")

    def on_test_epoch_start(self) -> None:
        self._start_evaluation("test")

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> dict:
        if "val" in self.evaluation:
            return self._evaluation_step(batch, batch_idx, "val", dataloader_idx)
        _, embeddings = self._step(batch, "val")
        return {
            "embeddings": apply_to_collection(
                embeddings, torch.Tensor, torch.Tensor.detach
            )
        }

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if "test" in self.evaluation:
            return self._evaluation_step(batch, batch_idx, "test", dataloader_idx)
        return self._step(batch, "test")[0]

    def on_validation_epoch_end(self) -> None:
        self._finish_evaluation("val")

    def on_test_epoch_end(self) -> None:
        self._finish_evaluation("test")
