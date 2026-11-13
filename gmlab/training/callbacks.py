"""Lifecycle consumers; optimization and required synchronization stay in PL."""

import json
import math
import os
import time
from collections.abc import Callable, Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
from lightning_utilities.core.apply_func import apply_to_collection
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..utils.io import atomic_write
from .run_log import RunLogger
from .tracking import media_logger

# Preserve useful existing theme names/colors without importing the old src.
THEMES = {
    "default": ("bold #fb8500", "#2ec4b6", "green1"),
    "cool_modern": ("bold #4A90E2", "#D1E8E2", "#4A90E2"),
    "vibrant_high_contrast": ("bold #FF6347", "#FFFFFF", "#FF6347"),
    "magenta": ("bold #FF69B4", "grey70", "#FF1493"),
    "green_burgundy": ("bold #556B2F", "grey70", "#8B0000"),
}


class DataSummary(Callback):
    """Display explicitly supplied data/config facts once per fit, on rank zero."""

    def __init__(
        self,
        facts: Mapping[str, Any],
        theme: str = "default",
        run_log: RunLogger | None = None,
    ) -> None:
        self.facts = dict(facts)
        self.theme = theme
        self.run_log = run_log

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            title, text, _ = THEMES[self.theme]
            table = Table(show_header=False, box=None)
            for key, value in self.facts.items():
                table.add_row(str(key), str(value), style=text)
            panel = Panel(table, title="Data", title_align="left", style=title)
            if self.run_log is None:
                Console().print(panel)
            else:
                self.run_log.print_table(panel)


class ValidationSummary(Callback):
    """Print complete epoch metrics once on rank zero after evaluation."""

    def __init__(
        self,
        run_log: RunLogger | None = None,
        step_console_interval: int = 50,
    ) -> None:
        if (
            isinstance(step_console_interval, bool)
            or not isinstance(step_console_interval, Integral)
            or step_console_interval < 1
        ):
            raise ValueError("step_console_interval must be a positive integer")
        self.run_log = run_log
        self.step_console_interval = int(step_console_interval)

    @staticmethod
    def _scalar_metrics(values, predicate) -> dict[str, float]:
        selected = [
            (name, value.detach().reshape(()))
            for name, value in values.items()
            if torch.is_tensor(value) and value.numel() == 1 and predicate(name)
        ]
        if not selected:
            return {}
        packed = torch.stack([value for _, value in selected]).cpu().tolist()
        return {name: float(value) for (name, _), value in zip(selected, packed)}

    def on_fit_start(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or self.run_log is None:
            return
        parameters = tuple(pl_module.model.parameters())
        total = sum(parameter.numel() for parameter in parameters)
        trainable = sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        )
        self.run_log.info(
            "Trainer ready: "
            f"strategy={type(trainer.strategy).__name__}, "
            f"world_size={trainer.world_size}, device={pl_module.device}, "
            f"precision={trainer.precision}, "
            f"epochs={trainer.max_epochs}, "
            f"batches_per_rank={trainer.num_training_batches}, "
            f"estimated_steps={trainer.estimated_stepping_batches}, "
            f"visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
        self.run_log.info(
            f"Model parameters: total={total:,}, trainable={trainable:,}, "
            f"frozen={total - trainable:,}"
        )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or self.run_log is None:
            return
        train_loader = trainer.train_dataloader
        if getattr(train_loader, "flattened", None):
            train_loader = train_loader.flattened[0]
        budgets = getattr(train_loader, "budgets", None)
        if budgets is not None:
            total = sum(budgets.values())
            self.run_log.info(
                "Training source batches per rank: "
                + ", ".join(
                    f"{name}={count} ({count / total:.1%})"
                    for name, count in budgets.items()
                )
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not trainer.is_global_zero or self.run_log is None:
            return
        if (
            batch_idx + 1
        ) % self.step_console_interval and batch_idx + 1 < trainer.num_training_batches:
            return
        metrics = self._scalar_metrics(
            trainer.callback_metrics,
            lambda name: name.startswith("train/") and name.endswith("_step"),
        )
        source = (
            batch.get("source") if getattr(pl_module, "named_training", False) else None
        )
        if source is not None:
            metrics = dict(pl_module.current_step_metrics)
        for optimizer_index, optimizer in enumerate(trainer.optimizers):
            for group_index, group in enumerate(optimizer.param_groups):
                fallback = f"optimizer_{optimizer_index}_group_{group_index}"
                name = group.get("name", fallback)
                metrics[f"lr/{name}"] = float(group["lr"])
        if not metrics:
            return
        message = (
            f"Training step epoch={trainer.current_epoch + 1} "
            f"batch={batch_idx + 1}/{trainer.num_training_batches} "
            f"source={source or 'default'} "
            f"step={trainer.global_step}: {RunLogger.scalar_metrics(metrics)}"
        )
        self.run_log.info(message)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # Lightning computes sync_dist epoch metrics lazily on access. Every
        # rank must materialize them before rank-zero rendering, otherwise a
        # later callback's collectives can collide with these reductions.
        epoch_metrics = trainer.callback_metrics
        if not trainer.is_global_zero:
            return
        metrics = self._scalar_metrics(
            epoch_metrics,
            lambda name: name.startswith("train/") and name.endswith("_epoch"),
        )
        if metrics:
            message = RunLogger.scalar_metrics(metrics)
            if self.run_log is None:
                Console().print(
                    f"Training epoch={trainer.current_epoch + 1} "
                    f"step={trainer.global_step}: {message}"
                )
            else:
                self.run_log.info(
                    f"Training epoch={trainer.current_epoch + 1} "
                    f"step={trainer.global_step}: {message}"
                )

    def on_validation_end(self, trainer, pl_module) -> None:
        if (
            trainer.sanity_checking
            or trainer.fast_dev_run
            or not trainer.is_global_zero
        ):
            return
        evaluation = getattr(pl_module, "evaluation_results", {}).get("val")
        metrics = (
            evaluation["metrics"]
            if evaluation is not None
            else {
                name.removeprefix("val/"): value
                for name, value in self._scalar_metrics(
                    trainer.callback_metrics,
                    lambda name: name.startswith("val/"),
                ).items()
            }
        )
        if not metrics:
            return
        epoch, step = trainer.current_epoch + 1, trainer.global_step
        table = Table(
            title=f"Validation · epoch {epoch} · step {step}",
            header_style="bold cyan",
            border_style="dim",
            min_width=48,
        )
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for name, value in metrics.items():
            table.add_row(name, f"{value:.4f}")
        if self.run_log is None:
            Console().print(table)
        else:
            self.run_log.print_table(table)


class TrainingDiagnostics(Callback):
    """Record timing and native callback state without changing training math."""

    def __init__(self, run_log: RunLogger, interval: int = 50) -> None:
        self.run_log = run_log
        self.interval = max(1, int(interval))
        self._batch_end = None
        self._batch_start = None
        self._data_seconds = None
        self._checkpoint_paths: dict[int, tuple[str, str]] = {}
        self._early_stopping_states: dict[int, tuple[int, str]] = {}

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._batch_end = None
        self._data_seconds = None

    def on_validation_start(self, trainer, pl_module) -> None:
        self._batch_end = None
        self._data_seconds = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        now = time.perf_counter()
        if self._batch_end is not None:
            self._data_seconds = now - self._batch_end
        self._batch_start = now

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        if not trainer.is_global_zero or (trainer.global_step + 1) % self.interval:
            return
        norms = [
            torch.linalg.vector_norm(parameter.grad.detach())
            for parameter in pl_module.parameters()
            if parameter.grad is not None
        ]
        if norms:
            norm = torch.linalg.vector_norm(torch.stack(norms)).item()
            self.run_log.info(
                f"Gradient step={trainer.global_step + 1}: norm_l2={norm:.4f}",
                console=True,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        now = time.perf_counter()
        report_batch = (
            batch_idx + 1
        ) % self.interval == 0 or batch_idx + 1 >= trainer.num_training_batches
        if trainer.is_global_zero and self._batch_start is not None and report_batch:
            details = [f"step_seconds={now - self._batch_start:.4f}"]
            if self._data_seconds is not None:
                details.append(f"data_seconds={self._data_seconds:.4f}")
            self.run_log.info(
                f"Timing epoch={trainer.current_epoch + 1} "
                f"batch={batch_idx + 1} step={trainer.global_step}: "
                + ", ".join(details),
                console=True,
            )
        self._batch_end = now

    def _native_state(self, trainer) -> None:
        if not trainer.is_global_zero:
            return
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                paths = (callback.best_model_path, callback.last_model_path)
                key = id(callback)
                if paths != self._checkpoint_paths.get(key):
                    for path in set(paths) - set(self._checkpoint_paths.get(key, ())):
                        if path and Path(path).is_file():
                            self.run_log.info(f"Checkpoint saved: {path}")
                    self._checkpoint_paths[key] = paths
            elif isinstance(callback, EarlyStopping):
                state = (callback.wait_count, str(callback.best_score))
                if state != self._early_stopping_states.get(id(callback)):
                    self.run_log.info(
                        f"EarlyStopping monitor={callback.monitor} "
                        f"wait={callback.wait_count}/{callback.patience} "
                        f"best={callback.best_score}",
                        console=True,
                    )
                    self._early_stopping_states[id(callback)] = state

    def on_validation_end(self, trainer, pl_module) -> None:
        if not trainer.sanity_checking:
            self._native_state(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._native_state(trainer)
        if trainer.is_global_zero and pl_module.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(pl_module.device) / 2**30
            self.run_log.info(f"GPU peak allocated: {peak:.3f} GiB")


class RNGState(Callback):
    """Single-process model/Python/NumPy RNG at epoch-boundary checkpoints.

    Uses the installed Lightning Fabric RNG utilities rather than a second RNG
    implementation. Their private API is pinned by tests to PL/Fabric 2.6.1.
    This does not restore random Dataset workers or sampler position. Distributed
    per-rank RNG recovery is deliberately rejected until such a contract exists.
    """

    def __init__(self):
        self._pending_rng_state = None

    def on_fit_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError("RNGState currently supports single-process recovery only")

    def state_dict(self):
        states = _collect_rng_states(include_cuda=torch.cuda.is_initialized())
        if "numpy" in states:
            algorithm, values, position, has_gauss, cached = states["numpy"]
            # Keep PL's RNG semantics, but use safe builtins in torch checkpoints.
            states["numpy"] = (algorithm, values.tolist(), position, has_gauss, cached)
        return states

    def load_state_dict(self, state_dict):
        self._pending_rng_state = state_dict

    def on_train_start(self, trainer, pl_module):
        # Setup and sanity validation may consume RNG after checkpoint loading.
        if self._pending_rng_state is not None:
            _set_rng_states(self._pending_rng_state)
            self._pending_rng_state = None


class ValidationMedia(Callback):
    """Render once per selected validation epoch, reusing the step's embeddings.

    ``render(batch, outputs, max_items)`` returns (images, captions). It receives
    detached CPU tensors; images may be arrays, PIL images or tensors. Frequency
    and limit are explicit. Without a compatible logger, rendering is skipped.
    No batch or computation graph is cached across hooks. Optional source selects
    a declared retrieval source; named sources append their name to the media key.
    Without source selection, retain the first eligible batch per epoch.
    """

    def __init__(
        self,
        render: Callable | str,
        *,
        every_n_epochs: int,
        max_items: int,
        key: str = "val/images",
        source: str | None = None,
    ) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            for value in (every_n_epochs, max_items)
        ):
            raise ValueError("Media frequency and limit must be positive integers")
        if isinstance(render, str):
            from ..components import resolve_component

            render = resolve_component(render)
        if not callable(render):
            raise TypeError("Media render must be callable")
        self.render = render
        self.every_n_epochs = every_n_epochs
        self.max_items = max_items
        if source is not None and (
            not isinstance(source, str)
            or not source
            or source != source.strip()
            or "/" in source
        ):
            raise ValueError("Media source must be a declared evaluation source name")
        self.source = source
        self.key = key
        self.logged = False

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.source is not None:
            sources = getattr(pl_module, "evaluation", {}).get("val")
            if not isinstance(sources, Mapping) or self.source not in sources:
                raise ValueError(f"Unknown validation media source: {self.source}")
        self.logged = False

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (
            trainer.sanity_checking
            or not trainer.is_global_zero
            or self.logged
            or (trainer.current_epoch + 1) % self.every_n_epochs
        ):
            return
        resolve_source = getattr(pl_module, "evaluation_source_name", None)
        if self.source is not None and not callable(resolve_source):
            raise ValueError("Source-selective media requires named evaluation sources")
        source_name = (
            resolve_source("val", dataloader_idx) if callable(resolve_source) else None
        )
        if self.source is not None and source_name != self.source:
            return
        sinks = [media_logger(logger) for logger in trainer.loggers]
        sinks = [sink for sink in sinks if callable(getattr(sink, "log_image", None))]
        if not sinks:
            return

        def detach(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().cpu()

        with torch.no_grad():
            images, captions = self.render(
                apply_to_collection(batch, torch.Tensor, detach),
                apply_to_collection(outputs, torch.Tensor, detach),
                self.max_items,
            )
        if len(images) != len(captions):
            raise ValueError("Each media image needs one caption")
        images = apply_to_collection(images[: self.max_items], torch.Tensor, detach)
        captions = captions[: self.max_items]
        for sink in sinks:
            key = f"{self.key}/{source_name}" if source_name is not None else self.key
            sink.log_image(key, images, caption=captions, step=trainer.global_step)
        self.logged = True


def _tracking_identity(trainer):
    if not trainer.loggers:
        return None
    logger = media_logger(trainer.loggers[0])
    # SwanLabLogger.version accesses .experiment and can start another run after
    # finalize(failed). Read the existing object without initializing anything.
    experiment = getattr(logger, "_experiment", None)
    run_id = getattr(experiment, "id", None)
    if run_id is not None:
        return run_id
    # Older SDKs expose the identity through the existing public run record.
    public = getattr(experiment, "public", None)
    return getattr(public, "run_id", None)


class FinalValidation(Callback):
    """Force the final training boundary through Lightning's native val loop.

    The Trainer schedule fields are changed only at a stopping boundary,
    before PL decides to run validation. Native checkpoint/early-stop callbacks
    then consume that result in their normal order. Tests pin this to PL 2.6.1.
    No nested Trainer.validate call or private checkpoint-save API is used.
    """

    def __init__(self):
        self._original_schedule = None

    def on_train_start(self, trainer, pl_module):
        self._original_schedule = (
            trainer.check_val_every_n_epoch,
            trainer.val_check_batch,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        final_epoch = (
            trainer.max_epochs is not None
            and trainer.max_epochs >= 0
            and trainer.current_epoch + 1 >= trainer.max_epochs
        )
        epoch_end = batch_idx + 1 >= trainer.num_training_batches
        final_step = trainer.max_steps >= 0 and trainer.global_step >= trainer.max_steps
        if (final_epoch and epoch_end) or final_step or trainer.should_stop:
            trainer.check_val_every_n_epoch = 1
            trainer.val_check_batch = 1

    def on_fit_end(self, trainer, pl_module):
        self._restore_schedule(trainer)

    def on_exception(self, trainer, pl_module, exception):
        self._restore_schedule(trainer)

    def _restore_schedule(self, trainer):
        if self._original_schedule is not None:
            epoch_interval, batch_interval = self._original_schedule
            trainer.check_val_every_n_epoch = epoch_interval
            trainer.val_check_batch = batch_interval
            self._original_schedule = None


class MonitorGuard(Callback):
    """Reject absent/nonfinite checkpoint monitors before checkpoint consumption."""

    def __init__(self, monitor: str) -> None:
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.fast_dev_run:
            return
        evaluation = getattr(pl_module, "evaluation_results", {}).get("val")
        if "val" in getattr(pl_module, "evaluation", {}) and (
            evaluation is None
            or evaluation["global_step"] != trainer.global_step
            or (
                self.monitor.startswith("val/")
                and self.monitor[4:] not in evaluation["metrics"]
            )
        ):
            raise ValueError(
                f"Checkpoint monitor lacks a complete current evaluation: {self.monitor}"
            )
        value = trainer.callback_metrics.get(self.monitor)
        if value is None or not torch.as_tensor(value).isfinite().all():
            raise ValueError(f"Missing or non-finite monitor: {self.monitor}")


class RunManifest(Callback):
    """Consume completed validation results and write an atomic local run record.

    Lifecycle status is finalized by the entrypoint after fit and checkpoint
    callbacks finish. Sanity/pre-fit validation never becomes a final result.
    Named state keys allow separately named instances without checkpoint clashes.
    """

    def __init__(
        self, path: str | Path, metadata: Mapping[str, Any], *, name: str = "run"
    ) -> None:
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.name = name
        self.validation = None
        self.validation_count = 0
        self.fitting = False
        self.run_id = None
        self.parent_run_id = None
        self.end_step = None
        self.test_result = None
        self.last_checkpoint_owner = None

    @property
    def state_key(self):
        return f"{type(self).__qualname__}[{self.name}]"

    def state_dict(self):
        return {
            "validation": self.validation,
            "validation_count": self.validation_count,
            "swanlab_run_id": self.run_id,
        }

    def load_state_dict(self, state_dict):
        self.validation = state_dict["validation"]
        self.validation_count = state_dict["validation_count"]
        self.parent_run_id = state_dict.get("swanlab_run_id")

    def capture_report_state(self) -> dict:
        """Preserve the training report before PL restores a test checkpoint."""
        return {
            "validation": self.validation,
            "validation_count": self.validation_count,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
        }

    def restore_report_state(self, state: dict) -> None:
        """Restore training provenance after a selected-weight test, even on failure."""
        self.validation = state["validation"]
        self.validation_count = state["validation_count"]
        self.run_id = state["run_id"]
        self.parent_run_id = state["parent_run_id"]

    def on_fit_start(self, trainer, pl_module):
        self.fitting = True
        self.end_step = None
        self.test_result = None
        self.validation = None  # A restored metric is not a fresh final result.
        self.write(trainer, "running")

    def on_validation_end(self, trainer, pl_module):
        if not self.fitting or trainer.sanity_checking:
            return
        evaluation = getattr(pl_module, "evaluation_results", {}).get("val")
        if "val" in getattr(pl_module, "evaluation", {}):
            if evaluation is None or evaluation["global_step"] != trainer.global_step:
                return
            metrics = {
                f"val/{name}": value for name, value in evaluation["metrics"].items()
            }
        else:
            metrics = {
                key: value.detach().cpu().item()
                for key, value in trainer.callback_metrics.items()
                if key.startswith("val/")
                and torch.is_tensor(value)
                and value.numel() == 1
            }
        if metrics:
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError("Non-finite validation metrics cannot complete a run")
            self.run_id = _tracking_identity(trainer) or self.run_id
            self.validation_count += 1
            self.validation = {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "metrics": metrics,
                "evaluation": evaluation,
            }

    def on_train_start(self, trainer, pl_module):
        self.run_id = _tracking_identity(trainer) or self.run_id

    def on_fit_end(self, trainer, pl_module):
        self.fitting = False

    def write(self, trainer, status, error=None):
        if not trainer.is_global_zero:
            return
        self.run_id = _tracking_identity(trainer) or self.run_id
        checkpoint = trainer.checkpoint_callback
        last = (
            self.last_checkpoint_owner.last_model_path
            if self.last_checkpoint_owner is not None
            else getattr(checkpoint, "last_model_path", None)
        )
        record = {
            **self.metadata,
            "status": status,
            "global_step": (
                self.end_step if self.end_step is not None else trainer.global_step
            ),
            "epoch": trainer.current_epoch,
            "validation_count": self.validation_count,
            "latest_validation": self.validation,
            "final_metrics": (
                self.validation["metrics"]
                if status == "completed"
                and self.validation is not None
                and self.validation["global_step"] == trainer.global_step
                else None
            ),
            "checkpoint": last or None,
            "best_checkpoint": getattr(checkpoint, "best_model_path", None) or None,
            "best_score": (
                float(checkpoint.best_model_score)
                if getattr(checkpoint, "best_model_score", None) is not None
                else None
            ),
            "test": self.test_result,
            "swanlab_run_id": self.run_id,
            "parent_swanlab_run_id": self.parent_run_id,
            "error": error,
        }
        contents = json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n"
        atomic_write(self.path, contents.encode("utf-8"))


def standard_callbacks(
    run_dir: str | Path,
    *,
    logger_enabled: bool,
    progress: bool,
    summary: bool,
    theme: str,
    data_facts: Mapping[str, Any] | None,
    run_log: RunLogger | None = None,
    step_console_interval: int = 50,
    checkpoint_options: Mapping[str, Any] | None = None,
) -> list[Callback]:
    """Configure native PL callbacks; no custom manager or hook dispatcher."""
    title, text, bar = THEMES[theme]
    options = {
        "monitor": "val/loss",
        "mode": "min",
        "save_last": True,
        "save_top_k": 1,
        "filename": "epoch={epoch}-step={step}",
        "auto_insert_metric_name": False,
        "verbose": True,
        **(checkpoint_options or {}),
    }
    if "dirpath" in options:
        raise ValueError("Checkpoint directory is owned by run_dir")
    rolling_last = bool(options.get("save_last")) and options.get("monitor") is not None
    if rolling_last and options.get("save_last") == "link":
        raise ValueError(
            "Monitored checkpoints require save_last=True for an independent latest "
            "snapshot; save_last='link' can point to an older best checkpoint"
        )
    if rolling_last and options.get("filename") == "last":
        raise ValueError(
            "Checkpoint filename 'last' is reserved for the latest snapshot"
        )
    if rolling_last:
        # PL 2.6.1 refreshes save_last only after a successful top-k save. A
        # separate unmonitored native callback keeps last current on plateaus.
        options["save_last"] = False
    callbacks = [ModelCheckpoint(dirpath=Path(run_dir) / "checkpoints", **options)]
    callbacks.append(ValidationSummary(run_log, step_console_interval))
    if rolling_last:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(run_dir) / "checkpoints",
                monitor=None,
                save_top_k=1,
                # Native exception saves do not populate best_model_path.
                # save_last registers the path before serializing callback state.
                save_last=True,
                verbose=True,
                save_on_exception=options.get("save_on_exception", False),
                filename="last",
                enable_version_counter=False,
                auto_insert_metric_name=False,
                every_n_epochs=options.get("every_n_epochs"),
                every_n_train_steps=options.get("every_n_train_steps"),
                train_time_interval=options.get("train_time_interval"),
                save_on_train_epoch_end=options.get("save_on_train_epoch_end"),
            )
        )
    if logger_enabled:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if progress:
        callbacks.append(
            RichProgressBar(
                leave=False,
                theme=RichProgressBarTheme(
                    description=title, progress_bar=bar, metrics=text
                ),
            )
        )
    if summary:
        callbacks.append(RichModelSummary(max_depth=1, header_style=title))
        if data_facts:
            callbacks.append(DataSummary(data_facts, theme, run_log))
    return callbacks
