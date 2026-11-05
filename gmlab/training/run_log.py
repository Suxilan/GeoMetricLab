"""Per-run text logging with scoped library and terminal integration."""

import hashlib
import inspect
import logging
import os
import re
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.table import Table


LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_active_log: ContextVar["RunLogger | None"] = ContextVar("gmlab_run_log", default=None)
_ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def current_run_log() -> "RunLogger | None":
    """Return the current run's text sink, if one is active."""
    return _active_log.get()


class _LibraryRecords(logging.Handler):
    """Forward library records to the run file; leave existing handlers intact."""

    def __init__(self, owner: "RunLogger") -> None:
        super().__init__(logging.DEBUG)
        self.owner = owner

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.WARNING or (
            rank_zero_only.rank != 0 and record.levelno < logging.ERROR
        ):
            return
        try:
            self.owner._write(record.levelname, f"{record.name}: {self.format(record)}")
        except OSError:
            self.handleError(record)


class _ConsoleStream:
    """Mirror complete terminal lines while retaining the original stream."""

    def __init__(self, stream, owner: "RunLogger", *, is_stderr: bool) -> None:
        self.stream = stream
        self.owner = owner
        self.is_stderr = is_stderr
        self.pending = ""
        self.active = True
        self._traceback = False

    def _base_stream(self):
        stream = self.stream
        seen = {id(self)}
        while isinstance(stream, _ConsoleStream):
            if id(stream) in seen:
                raise RuntimeError("Cyclic console stream wrappers")
            seen.add(id(stream))
            stream = stream.stream
        return stream

    def __getattr__(self, name):
        return getattr(self.stream, name)

    @staticmethod
    def _from_library_handler() -> bool:
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if (
                    frame.f_code.co_name == "emit"
                    and frame.f_globals.get("__name__") == "logging"
                ):
                    return True
                frame = frame.f_back
            return False
        finally:
            del frame

    def write(self, value: str) -> int:
        result = self._base_stream().write(value)
        if not self.active:
            return result
        if self._from_library_handler():
            return result
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            clean = _ansi.sub("", line)
            if (
                not clean.strip()
                or "\r" in line
                or self.owner._is_warning_line(clean.lstrip())
            ):
                continue
            if rank_zero_only.rank != 0:
                if not self.is_stderr:
                    continue
                self._traceback |= (
                    "Traceback" in clean or "Error:" in clean or "Exception:" in clean
                )
                if not self._traceback:
                    continue
            self.owner._write("CONSOLE", clean)
        return result

    def flush(self) -> None:
        self._base_stream().flush()


class RunLogger:
    """Own one run file and capture explicit, library, and terminal events.

    Warning records remain on the terminal. Rank and DataLoader worker identity
    prefix every file line. Appends are one write per line to prevent interleaving.
    """

    def __init__(self, path: str | Path, level: str = "INFO") -> None:
        normalized = level.upper()
        if normalized not in LEVELS:
            raise ValueError(f"Unknown run log level: {level!r}")
        self.path = Path(path)
        self.level = LEVELS[normalized]
        self._entered = False
        self._bridges: list[tuple[logging.Logger, int, logging.Handler | None]] = []
        self._streams = None
        self._mirrors = None
        self._token = None
        self.console = Console(markup=False, highlight=False)

    def __enter__(self) -> "RunLogger":
        if self._entered:
            raise RuntimeError("RunLogger is already active")
        if current_run_log() is not None:
            raise RuntimeError("A different RunLogger is already active")
        self._entered = True
        self._streams = (sys.stdout, sys.stderr)
        for name in ("", "gmlab", "pytorch_lightning", "lightning", "lightning_fabric"):
            logger = logging.getLogger(name)
            if name == "gmlab":
                self._bridges.append((logger, logger.level, None))
                logger.setLevel(logging.INFO)
                continue
            if name and name != "gmlab" and logger.propagate:
                continue
            handler = _LibraryRecords(self)
            self._bridges.append((logger, logger.level, handler))
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
        self._mirrors = (
            _ConsoleStream(sys.stdout, self, is_stderr=False),
            _ConsoleStream(sys.stderr, self, is_stderr=True),
        )
        sys.stdout, sys.stderr = self._mirrors
        self.console = Console(file=sys.stdout, markup=False, highlight=False)
        self._token = _active_log.set(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if not self._entered:
            return
        sys.stdout, sys.stderr = self._streams
        for mirror, fallback in zip(self._mirrors, (sys.__stdout__, sys.__stderr__)):
            mirror.active = False
            # Library handlers may retain a stream after the run has ended.
            mirror.stream = fallback
        for logger, level, handler in reversed(self._bridges):
            if handler is not None:
                logger.removeHandler(handler)
                handler.close()
            logger.setLevel(level)
        _active_log.reset(self._token)
        self._entered = False

    @staticmethod
    def _is_warning_line(line: str) -> bool:
        return line.startswith(("WARNING ", "WARNING:", "warning: ")) or bool(
            re.match(r"^\S+\.py:\d+: (?:\w*Warning|Warning):", line)
        )

    def environment(self) -> None:
        """Show a concise environment identity; full details live in run.json."""
        if rank_zero_only.rank != 0:
            return
        import torch

        from ..utils.provenance import code_version, dependency_versions

        code = code_version()
        hashes = code.pop("gmlab_sha256", {})
        code["gmlab_source_digest"] = hashlib.sha256(
            repr(sorted(hashes.items())).encode()
        ).hexdigest()
        versions = dependency_versions()
        self.info(
            "Environment: "
            f"Python {sys.version.split()[0]}, torch {versions.get('torch')}, "
            f"Lightning {versions.get('pytorch-lightning')}, "
            f"CUDA {torch.version.cuda}, GPUs {torch.cuda.device_count()}, "
            f"git {str(code.get('commit', 'unknown'))[:12]}"
            f"{' dirty' if code.get('dirty') else ''}"
        )
        self.info(
            f"Command: {' '.join(sys.argv)}; "
            f"source digest: {code['gmlab_source_digest'][:16]}"
        )

    def _write(self, severity: str, message: str) -> None:
        from torch.utils.data import get_worker_info

        if rank_zero_only.rank != 0 and severity not in {
            "ERROR",
            "CRITICAL",
            "CONSOLE",
        }:
            return
        worker = get_worker_info()
        worker_name = "main" if worker is None else str(worker.id)
        prefix = f"[rank={rank_zero_only.rank} worker={worker_name}]"
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            for line in str(message).splitlines() or [""]:
                entry = (
                    line
                    if severity == "CONSOLE" and rank_zero_only.rank == 0
                    else f"{timestamp} {severity} {prefix} {line}"
                )
                os.write(descriptor, f"{entry}\n".encode())
        finally:
            os.close(descriptor)

    def emit(self, severity: str, message: str, *, console: bool | None = None) -> None:
        normalized = severity.upper()
        if normalized not in LEVELS:
            raise ValueError(f"Unknown run log severity: {severity!r}")
        show = LEVELS[normalized] >= self.level if console is None else console
        if show and (rank_zero_only.rank == 0 or normalized in {"ERROR", "CRITICAL"}):
            prefix = (
                "WARNING:"
                if normalized == "WARNING"
                else (f"[rank {rank_zero_only.rank}] {normalized}")
            )
            self.console.print(f"{prefix} {message}")
            if normalized != "WARNING" and (
                not self._entered or rank_zero_only.rank != 0
            ):
                self._write(normalized, message)
        elif normalized != "WARNING" and rank_zero_only.rank == 0:
            self._write(normalized, message)

    def debug(self, message: str, *, console: bool | None = None) -> None:
        self.emit("DEBUG", message, console=console)

    def info(self, message: str, *, console: bool | None = None) -> None:
        self.emit("INFO", message, console=console)

    def warning(self, message: str, *, console: bool | None = None) -> None:
        self.emit("WARNING", message, console=console)

    def error(self, message: str, *, console: bool | None = None) -> None:
        self.emit("ERROR", message, console=console)

    def exception(self, message: str) -> None:
        self.error(f"{message}\n{traceback.format_exc()}")

    def print_table(self, table: Table, *, console: bool = True) -> None:
        if rank_zero_only.rank != 0:
            return
        if console:
            self.console.print(table)
        else:
            from io import StringIO

            plain = StringIO()
            Console(file=plain, width=self.console.width).print(table)
            self._write("INFO", plain.getvalue().rstrip())

    @staticmethod
    def scalar_metrics(metrics: dict[str, Any]) -> str:
        return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
