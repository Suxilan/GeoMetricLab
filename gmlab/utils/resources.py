"""Optional sampled process-tree resource measurements for execution tools."""

import math
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .io import atomic_json


def _gpu_memory_mib(process_ids: set[int]) -> int | None:
    """Return this tree's reported GPU memory, or None when unavailable."""
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).splitlines()
        return sum(
            int(row.split(",")[1])
            for row in rows
            if row.strip() and int(row.split(",")[0]) in process_ids
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@contextmanager
def monitor_process(
    pid: int, output: str | Path | None, interval: float = 1.0
) -> Iterator[None]:
    """Write sampled peak resources for pid and its descendants on scope exit.

    None disables monitoring without importing psutil. PSS accounts for shared
    pages; summed RSS can double-count them. Measurements are sampled peaks, not
    allocator or operating-system high-water marks. Missing GPU tooling is
    recorded as unavailable. Sampling failures propagate rather than producing
    a silently incomplete successful report. psutil is an optional dependency.
    """
    if output is None:
        yield
        return
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Resource sampling interval must be finite and positive")
    import psutil

    parent = psutil.Process(pid)
    stop = threading.Event()
    errors = []
    result = dict(
        cpu_affinity=sorted(parent.cpu_affinity()),
        sample_interval_seconds=interval,
        peak_process_tree_pss_bytes=0,
        peak_process_tree_rss_sum_bytes=0,
        peak_gpu_process_memory_mib=0,
        peak_cpu_cores_sampled=0.0,
        samples=0,
        gpu_samples_available=False,
        scope=(
            "sampled process tree; PSS accounts shared pages, RSS sum may double count; "
            "GPU memory belongs to this process tree"
        ),
    )

    def sample():
        previous_cpu = {}
        previous_time = time.monotonic()
        try:
            while not stop.is_set():
                try:
                    processes = [parent, *parent.children(recursive=True)]
                except psutil.NoSuchProcess:
                    break
                rss = pss = cpu_delta = 0
                process_ids = set()
                for process in processes:
                    try:
                        memory = process.memory_full_info()
                        rss += memory.rss
                        pss += getattr(memory, "pss", 0)
                        cpu = process.cpu_times()
                        total = cpu.user + cpu.system
                        cpu_delta += max(
                            0, total - previous_cpu.get(process.pid, total)
                        )
                        previous_cpu[process.pid] = total
                        process_ids.add(process.pid)
                    except psutil.NoSuchProcess:
                        continue
                now = time.monotonic()
                result["peak_cpu_cores_sampled"] = max(
                    result["peak_cpu_cores_sampled"],
                    cpu_delta / max(now - previous_time, 1e-6),
                )
                previous_time = now
                result["peak_process_tree_pss_bytes"] = max(
                    result["peak_process_tree_pss_bytes"], pss
                )
                result["peak_process_tree_rss_sum_bytes"] = max(
                    result["peak_process_tree_rss_sum_bytes"], rss
                )
                gpu_memory = _gpu_memory_mib(process_ids)
                if gpu_memory is not None:
                    result["gpu_samples_available"] = True
                    result["peak_gpu_process_memory_mib"] = max(
                        result["peak_gpu_process_memory_mib"], gpu_memory
                    )
                result["samples"] += 1
                stop.wait(interval)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=5)
        if worker.is_alive():
            raise RuntimeError(
                "Resource sampler did not stop before report publication"
            )
        if errors:
            raise RuntimeError("Resource sampling failed") from errors[0]
        atomic_json(Path(output), result)
