"""One explicit entrypoint for native Lightning training and owned run resources."""

import copy
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger

from ..models.identity import model_identity_record
from ..utils.provenance import code_version, dependency_versions
from .callbacks import (
    FinalValidation,
    MonitorGuard,
    RunManifest,
    TrainingDiagnostics,
    standard_callbacks,
)
from .module import MetricModule
from .sources import MultiSourceLoader
from .run_log import RunLogger, current_run_log
from .tracking import (
    _BorrowedLogger,
    create_swanlab_logger,
    create_configured_logger,
    finish_owned_logger,
    initialize_owned_logger,
    is_swanlab_logger,
)


def _trainer_options(
    trainer_options: Mapping[str, Any],
    logger: Any,
    tracking: Mapping[str, Any] | None,
    final_validation: str,
) -> dict[str, Any]:
    if final_validation not in {"scheduled", "require", "force"}:
        raise ValueError("final_validation must be scheduled, require or force")
    options = dict(trainer_options)
    conflicts = {
        "logger",
        "callbacks",
        "default_root_dir",
        "enable_progress_bar",
        "enable_model_summary",
        "enable_checkpointing",
    } & options.keys()
    if conflicts:
        raise ValueError(f"Use explicit fit arguments for {sorted(conflicts)}")
    if (
        logger is not None
        and logger is not False
        and not isinstance(logger, (Logger, Mapping))
        and not is_swanlab_logger(logger)
    ):
        raise TypeError(
            "runtime.logger must be False, a Lightning Logger, or a component mapping"
        )
    if logger is not None and tracking is not None:
        raise ValueError("Choose injected logger/False or owned tracking options")
    return options


def _run_metadata(
    module: MetricModule,
    options: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    ckpt_path: str | Path | None,
    owns_logger: bool,
) -> dict[str, Any]:
    identity = model_identity_record(module.model)
    return {
        "code": code_version(),
        "dependencies": dependency_versions(),
        "metadata": dict(metadata or {}),
        "trainer_options": options,
        "resume_checkpoint": str(ckpt_path) if ckpt_path else None,
        "tracking_ownership": "entrypoint" if owns_logger else "caller/disabled",
        "constructed_model_sha256": identity["sha256"],
        "constructed_model_identity": identity,
        "components": {
            "model": repr(module.model),
            "criterion": repr(module.criterion),
            "optimizer": repr(module.optimizer_factory),
            "scheduler": repr(module.scheduler_factory),
            "training_sources": {
                name: {
                    "criterion": repr(source.criterion),
                    "batch_adapter": repr(source.batch_adapter),
                    "embedding_adapter": repr(source.embedding_adapter),
                }
                for name, source in module.training_sources.items()
            },
            "inference_batch_adapter": repr(module.inference_batch_adapter),
            "inference_embedding_adapter": repr(module.inference_embedding_adapter),
        },
    }


def _run_callbacks(
    run_dir: Path,
    manifest: RunManifest,
    callbacks: Sequence[pl.Callback],
    *,
    logger_enabled: bool,
    progress: bool,
    summary: bool,
    theme: str,
    data_facts: Mapping | None,
    run_log: RunLogger,
    step_console_interval: int,
    checkpoint_options: Mapping | None,
    final_validation: str,
) -> tuple[list[pl.Callback], ModelCheckpoint]:
    """Assemble native callbacks and select the checkpoint consumed by fit."""
    custom_callbacks = list(callbacks)
    configured_callbacks = standard_callbacks(
        run_dir,
        logger_enabled=logger_enabled,
        progress=progress,
        summary=summary,
        theme=theme,
        data_facts=data_facts,
        run_log=run_log,
        step_console_interval=step_console_interval,
        checkpoint_options=checkpoint_options,
    )
    native_checkpoints = [
        callback
        for callback in configured_callbacks
        if isinstance(callback, ModelCheckpoint)
    ]
    if any(isinstance(callback, ModelCheckpoint) for callback in custom_callbacks):
        configured_callbacks = [
            c for c in configured_callbacks if not isinstance(c, ModelCheckpoint)
        ]
    elif len(native_checkpoints) > 1:
        manifest.last_checkpoint_owner = native_checkpoints[1]
    configured_callbacks.extend(custom_callbacks)
    checkpoint = next(c for c in configured_callbacks if isinstance(c, ModelCheckpoint))
    if checkpoint.monitor:
        configured_callbacks.append(MonitorGuard(checkpoint.monitor))
    if final_validation == "force":
        configured_callbacks.append(FinalValidation())
    configured_callbacks.append(TrainingDiagnostics(run_log, step_console_interval))
    configured_callbacks.append(manifest)
    return configured_callbacks, checkpoint


def _require_final_validation(trainer: pl.Trainer, manifest: RunManifest) -> None:
    latest = manifest.validation
    if (
        not trainer.state.finished
        or not latest
        or latest["global_step"] != trainer.global_step
    ):
        raise RuntimeError("Run lacks a completed final-step validation result")
    manifest.end_step = trainer.global_step


def _test_selected_weights(
    trainer: pl.Trainer,
    module: MetricModule,
    loader: Iterable,
    checkpoint: ModelCheckpoint,
    manifest: RunManifest,
    selection: str,
) -> dict[str, Any]:
    """Test an explicit weight while retaining completed training provenance.

    PL restores callbacks from the selected checkpoint, including an older best.
    Restore checkpoint/report state even on failure; leave the tested model at
    its actual selected weight so its recorded digest is unambiguous.
    """
    if selection not in {"current", "best", "last"}:
        raise ValueError("test_weights must explicitly be current, best or last")
    selected_checkpoint = {
        "current": None,
        "best": checkpoint.best_model_path,
        "last": (
            manifest.last_checkpoint_owner.last_model_path
            if manifest.last_checkpoint_owner is not None
            else checkpoint.last_model_path
        ),
    }[selection]
    if selection != "current" and not selected_checkpoint:
        raise ValueError(f"No {selection} checkpoint exists")
    checkpoint_states = [
        (callback, copy.deepcopy(callback.state_dict()))
        for callback in trainer.checkpoint_callbacks
    ]
    manifest_state = manifest.capture_report_state()
    try:
        metrics = trainer.test(
            module, dataloaders=loader, ckpt_path=selected_checkpoint, verbose=False
        )
    finally:
        for callback, state in checkpoint_states:
            callback.load_state_dict(state)
        manifest.restore_report_state(manifest_state)
    evaluation = module.evaluation_results.get("test")
    identity = model_identity_record(module.model)
    return {
        "selection": selection,
        "checkpoint": selected_checkpoint,
        "weights_sha256": identity["sha256"],
        "model_identity": identity,
        "metrics": evaluation["metrics"] if evaluation is not None else metrics,
        **(
            {"sources": evaluation["sources"]}
            if evaluation and "sources" in evaluation
            else {}
        ),
    }


def _release_owned_logger(
    logger: Logger, trainer: pl.Trainer | None, status: str, *, swanlab: bool
) -> None:
    try:
        if swanlab:
            try:
                if getattr(logger, "_experiment", None) is not None:
                    logger.finalize(status)
            finally:
                finish_owned_logger(logger)
        else:
            logger.finalize(status)
    finally:
        if trainer is not None:
            trainer.loggers = []


def _prepare_run(
    run_dir: Path,
    checkpoint: str | Path | None,
    has_validation: bool,
    validate_before_fit: bool,
    final_validation: str,
    checkpoint_options: Mapping | None,
) -> Mapping | None:
    """Protect existing results and validate the selected validation/checkpoint policy."""
    if (run_dir / "run.json").exists() and checkpoint is None:
        raise FileExistsError(
            f"Run already exists: {run_dir}; choose a new directory or resume"
        )
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    if not has_validation:
        if validate_before_fit or final_validation != "scheduled":
            raise ValueError("The validation policy requires a validation loader")
        checkpoint_options = {
            "monitor": None,
            "save_top_k": 0,
            "save_last": True,
            **(checkpoint_options or {}),
        }
        monitor = checkpoint_options.get("monitor")
        if monitor and monitor.startswith("val/"):
            raise ValueError("A validation monitor requires a validation loader")
    return checkpoint_options


def _execute_training(
    trainer,
    module,
    train_loader,
    val_loader,
    manifest,
    checkpoint,
    *,
    ckpt_path,
    validate_before_fit,
    final_validation,
    test_loader,
    test_weights,
):
    """Run the selected validation, training and test phases in one logger lifetime."""
    if validate_before_fit:
        trainer.validate(
            module, dataloaders=val_loader, ckpt_path=ckpt_path, verbose=False
        )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )
    if not trainer.state.finished:
        raise RuntimeError("Training did not finish")
    if final_validation in {"require", "force"}:
        _require_final_validation(trainer, manifest)
    manifest.end_step = trainer.global_step
    if test_loader is not None:
        manifest.test_result = _test_selected_weights(
            trainer,
            module,
            test_loader,
            checkpoint,
            manifest,
            test_weights,
        )


def _log_run_setup(log, run_dir, options, metadata, train_loader, tracking):
    """Report component facts without changing the experiment or its lifecycle."""
    log.info(f"Run directory: {run_dir.resolve()}")
    log.info(f"Run manifest: {run_dir / 'run.json'}")
    log.info(f"Trainer options: {options}")
    resolved = (metadata or {}).get("resolved_config", {})
    recipe_runtime = resolved.get("runtime", {})
    if recipe_runtime:
        log.info(
            "Run policy: "
            f"seed={recipe_runtime.get('seed')}, "
            f"validate_before_fit={recipe_runtime.get('validate_before_fit')}, "
            f"final_validation={recipe_runtime.get('final_validation')}, "
            f"check_val_every_n_epoch={options.get('check_val_every_n_epoch')}"
        )
    loaders = (
        train_loader.loaders
        if isinstance(train_loader, MultiSourceLoader)
        else {"default": train_loader}
    )
    for name, loader in loaders.items():
        dataset = getattr(loader, "dataset", None)
        sampler = getattr(loader, "sampler", None)
        log.info(
            f"Training data source={name}: "
            f"dataset={type(dataset).__name__ if dataset is not None else 'unspecified'}, "
            f"batch_size_per_rank={getattr(loader, 'batch_size', None)}, "
            f"sampling_multiplier={getattr(dataset, 'sampling_multiplier', 1.0)}, "
            f"sampler={type(sampler).__name__ if sampler is not None else 'unspecified'}"
        )
    if tracking:
        tracking_summary = {
            key: tracking[key]
            for key in ("mode", "project", "experiment_name")
            if key in tracking
        }
        log.info(f"Experiment tracking: {tracking_summary}")


def fit(
    module: MetricModule,
    train_loader: Iterable,
    val_loader: Iterable | None = None,
    *,
    run_dir: str | Path,
    trainer_options: Mapping[str, Any],
    logger: Logger | Mapping[str, Any] | Literal[False] | None = None,
    tracking: Mapping[str, Any] | None = None,
    callbacks: Sequence[pl.Callback] = (),
    progress: bool = False,
    summary: bool = True,
    theme: str = "default",
    text_level: str = "INFO",
    data_facts: Mapping | None = None,
    metadata: Mapping | None = None,
    ckpt_path: str | Path | None = None,
    validate_before_fit: bool = False,
    test_loader: Iterable | None = None,
    test_weights: str = "current",
    checkpoint_options: Mapping | None = None,
    final_validation: str = "scheduled",
) -> pl.Trainer:
    """Train/validate with a native Trainer, returning it for test/predict use.

    ``logger=None`` creates an owned official SwanLabLogger (online unless
    tracking.mode is explicitly supplied). ``False`` disables recording;
    a component mapping creates an owned Lightning logger, finalized once at
    the end of this run. An injected logger instance is borrowed and never
    finalized by this entrypoint. Native hooks are forwarded in either case.
    Checkpoints are PL checkpoints; callers reconstruct the same module graph,
    optimizer/scheduler factories and callbacks before ``ckpt_path`` resume.

    ``trainer_options`` is passed to PL without inventing another Trainer config.
    Metadata should contain resolved config, seed and GT/evaluation protocol.
    Validation follows the experiment's policy. scheduled uses the native PL
    schedule and permits training without a validation loader. require checks
    for a fresh final-step result; force also schedules that result explicitly.
    ``checkpoint_options`` selects monitor/mode using native ModelCheckpoint.
    ``final_validation="force"`` adds a pinned PL schedule hook at stopping
    boundaries, so native callbacks see fresh final metrics. ``test_weights``
    explicitly selects current/best/last; test restore cannot replace the
    completed training record. Pre-validation/test share the logging lifetime.
    Owned loggers are detached when that lifetime ends so later Trainer calls
    cannot silently reopen their SDK run. Multi-source loaders replay their
    per-source cursor with deterministic samplers; worker-local random state
    is not recovered.
    """
    if test_weights not in {"current", "best", "last"}:
        raise ValueError("test_weights must explicitly be current, best or last")
    options = _trainer_options(trainer_options, logger, tracking, final_validation)
    if isinstance(train_loader, MultiSourceLoader):
        if options.get("use_distributed_sampler") is True:
            raise ValueError(
                "Multi-source loaders own rank sharding; set "
                "trainer.use_distributed_sampler=false"
            )
        options["use_distributed_sampler"] = False
    run_dir = Path(run_dir)
    owns_logger = logger is None or isinstance(logger, Mapping)
    owns_swanlab = logger is None
    checkpoint_options = _prepare_run(
        run_dir,
        ckpt_path,
        val_loader is not None,
        validate_before_fit,
        final_validation,
        checkpoint_options,
    )
    existing_log = current_run_log()
    if existing_log is not None and existing_log.path != run_dir / "train.log":
        raise ValueError("Active run log belongs to a different run directory")
    log_scope = (
        nullcontext(existing_log)
        if existing_log is not None
        else RunLogger(run_dir / "train.log", level=text_level)
    )
    with log_scope as log:
        if existing_log is None:
            log.environment()
        try:
            if owns_swanlab:
                tracking_options = {"logdir": str(run_dir), **dict(tracking or {})}
                active_logger = create_swanlab_logger(tracking_options)
            elif owns_logger:
                active_logger = create_configured_logger(logger)
            else:
                active_logger = logger
        except BaseException:
            log.exception("Experiment logger initialization failed")
            raise
        swanlab_lifetime = owns_logger and is_swanlab_logger(active_logger)
        _log_run_setup(log, run_dir, options, metadata, train_loader, tracking)
        trainer = None
        status = "failed"
        try:
            run_metadata = _run_metadata(
                module, options, metadata, ckpt_path, owns_logger
            )
            run_metadata["validation_policy"] = final_validation
            manifest = RunManifest(run_dir / "run.json", run_metadata)
            configured_callbacks, checkpoint = _run_callbacks(
                run_dir,
                manifest,
                callbacks,
                logger_enabled=active_logger is not False,
                progress=progress,
                summary=summary,
                theme=theme,
                data_facts=data_facts,
                run_log=log,
                step_console_interval=options.get("log_every_n_steps", 50),
                checkpoint_options=checkpoint_options,
                final_validation=final_validation,
            )
            trainer = pl.Trainer(
                **options,
                default_root_dir=run_dir,
                logger=_BorrowedLogger(active_logger)
                if active_logger is not False
                else False,
                callbacks=configured_callbacks,
                enable_progress_bar=progress,
                enable_model_summary=False,
                enable_checkpointing=True,
            )
            if swanlab_lifetime:
                initialize_owned_logger(active_logger)
            if active_logger is not False:
                active_logger.log_hyperparams(run_metadata)
            log.info("Starting metric training")
            _execute_training(
                trainer,
                module,
                train_loader,
                val_loader,
                manifest,
                checkpoint,
                ckpt_path=ckpt_path,
                validate_before_fit=validate_before_fit,
                final_validation=final_validation,
                test_loader=test_loader,
                test_weights=test_weights,
            )
            manifest.write(trainer, "completed")
            log.info(f"Completed metric training at step {trainer.global_step}")
            status = "success"
            return trainer
        except BaseException as error:
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
            status = "interrupted" if interrupted else "failed"
            if trainer is not None:
                manifest.write(trainer, status, f"{type(error).__name__}: {error}")
            log.exception("Metric run failed")
            raise
        finally:
            if owns_logger:
                if status == "success":
                    _release_owned_logger(
                        active_logger, trainer, status, swanlab=swanlab_lifetime
                    )
                else:
                    try:
                        _release_owned_logger(
                            active_logger, trainer, status, swanlab=swanlab_lifetime
                        )
                    except Exception:
                        log.exception(
                            "Logger finalization failed while handling the run failure"
                        )
