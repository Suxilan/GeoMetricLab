"""Logger ownership and scoped console configuration at the run boundary."""

import random
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.utilities.seed import isolate_rng

from ..components import build_component


def _current_swanlab_run(swanlab):
    """Return the active run across SwanLab versions that signal no run differently."""
    try:
        return swanlab.get_run()
    except (AttributeError, RuntimeError):
        return None


class _BorrowedLogger(Logger):
    """Let Trainer borrow a logger without taking ownership of its lifetime.

    The owner is either runtime.fit for a configured logger or the Python caller
    for an injected instance. Suppress per-stage finalization across validation,
    fit and test while forwarding the native recording and checkpoint contract.
    """

    def __init__(self, logger):
        super().__init__()
        self.wrapped = logger

    @property
    def name(self):
        return self.wrapped.name

    @property
    def version(self):
        return self.wrapped.version

    @property
    def save_dir(self):
        return self.wrapped.save_dir

    @property
    def root_dir(self):
        return self.wrapped.root_dir

    @property
    def log_dir(self):
        return self.wrapped.log_dir

    @property
    def group_separator(self):
        return self.wrapped.group_separator

    @property
    def experiment(self):
        return self.wrapped.experiment

    def log_graph(self, model, input_array=None):
        self.wrapped.log_graph(model, input_array=input_array)

    def after_save_checkpoint(self, checkpoint_callback):
        self.wrapped.after_save_checkpoint(checkpoint_callback)

    def log_metrics(self, metrics, step=None):
        self.wrapped.log_metrics(metrics, step=step)

    def log_hyperparams(self, params, *args, **kwargs):
        self.wrapped.log_hyperparams(params, *args, **kwargs)

    def save(self):
        self.wrapped.save()

    def finalize(self, status):
        pass  # The caller owns both successful and failed experiment closure.


def media_logger(logger: Logger) -> Logger:
    """Expose the underlying media sink while preserving its caller ownership."""
    return logger.wrapped if isinstance(logger, _BorrowedLogger) else logger


@contextmanager
def _isolate_logger_rng() -> Iterator[None]:
    """Restore training RNG even when logger setup raises.

    PL 2.6.1 isolate_rng restores after yield without a finally block. Defer the
    exception until its normal exit so both import and SDK failures restore RNG.
    """
    failure = None
    with isolate_rng(include_cuda=False):
        try:
            yield
        except BaseException as error:
            failure = error
    if failure is not None:
        raise failure


def is_swanlab_logger(logger: Logger) -> bool:
    """Recognize the actual official type without importing an unused SDK.

    Constructing a SwanLab logger has already loaded its integration module.
    Generic YAML loggers must not acquire a SwanLab dependency merely to close.
    """
    integration = sys.modules.get("swanlab.integration.pytorch_lightning")
    logger_type = getattr(integration, "SwanLabLogger", None)
    return logger_type is not None and isinstance(logger, logger_type)


def create_configured_logger(specification: Mapping[str, Any]) -> Logger:
    """Construct one owned Lightning logger from an explicit component mapping.

    Import/construction randomness is isolated from the training stream. Logger
    experiment initialization remains native and lazy; no SDK is presumed.
    """
    with _isolate_logger_rng():
        swanlab = sys.modules.get("swanlab")
        existing_run = _current_swanlab_run(swanlab) if swanlab is not None else None
        try:
            logger = build_component(specification)
        except (TypeError, ValueError, ImportError, AttributeError) as error:
            raise ValueError(f"runtime.logger: {error}") from error
    if not isinstance(logger, Logger):
        raise TypeError("runtime.logger must construct a Lightning Logger")
    if is_swanlab_logger(logger) and existing_run is not None:
        raise RuntimeError(
            "An existing SwanLab run is active; inject its logger explicitly"
        )
    return logger


def create_swanlab_logger(options: Mapping[str, Any]) -> Logger:
    """Create an owned official logger; online by default, no silent fallback."""
    # SDK imports can consume Python/NumPy RNG before experiment initialization.
    with _isolate_logger_rng():
        import swanlab
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        if rank_zero_only.rank == 0 and _current_swanlab_run(swanlab) is not None:
            raise RuntimeError(
                "An existing SwanLab run is active; inject its logger explicitly"
            )
        return SwanLabLogger(**({"mode": "online"} | dict(options)))


def finish_owned_logger(logger: Logger) -> None:
    """Close only this logger's still-active run, once, on the recording rank."""
    if rank_zero_only.rank != 0:
        return
    import swanlab

    current = _current_swanlab_run(swanlab)
    if current is not None and current is getattr(logger, "_experiment", None):
        swanlab.finish()


def initialize_owned_logger(logger: Logger) -> None:
    """Isolate SDK randomness from training and seeded offline run identities."""
    if rank_zero_only.rank != 0:
        return

    # SwanLab 0.7.11 generates offline IDs with Python random. Re-seeding an
    # experiment otherwise repeats IDs. Lightning's isolation preserves the
    # training RNG while OS entropy supplies an independent experiment identity.
    with _isolate_logger_rng():
        random.seed()
        logger.experiment
