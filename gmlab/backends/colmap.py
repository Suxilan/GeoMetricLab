"""COLMAP executable discovery, isolated adapter builds and explicit subprocesses.

No model, dataset, artifact lifecycle or benchmark resource policy lives here.
Prepared-adapter source assets are supplied by the caller, not located via tools.
"""

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from ..utils.io import atomic_json


def colmap_path(value=None):
    """Resolve an explicit executable or COLMAP on PATH; never guess a user home."""
    executable = value or shutil.which("colmap")
    if executable is None or not Path(executable).is_file():
        raise FileNotFoundError(f"COLMAP executable not found: {executable}")
    return str(Path(executable).resolve())


def sift_command(
    executable,
    *,
    database,
    image_root,
    image_list,
    max_image_size,
    max_features,
    gpu,
    threads,
):
    """Build the original-resolution COLMAP L1_ROOT SIFT command, without running it.

    max_image_size must be the maximum decoded input side when no resize is
    intended. Dataset discovery and extraction records belong to the caller.
    """
    if min(max_image_size, max_features, threads) < 1 or gpu < 0:
        raise ValueError("Invalid SIFT image/feature/thread count or GPU index")
    options = {
        "database_path": database,
        "image_path": image_root,
        "image_list_path": image_list,
        "FeatureExtraction.max_image_size": max_image_size,
        "FeatureExtraction.use_gpu": 1,
        "FeatureExtraction.gpu_index": gpu,
        "FeatureExtraction.num_threads": threads,
        "SiftExtraction.max_num_features": max_features,
        "SiftExtraction.first_octave": 0,
        "descriptor_normalization": "L1_ROOT",
        "log_target": "stderr",
    }
    return [str(executable), "feature_extractor", *_command_options(options)]


def prepared_bow_command(
    executable,
    *,
    database,
    prepared,
    threads,
    topk,
    repeats,
):
    """Build common prepared-BoW flags; the caller selects encode/build/query.

    Query parallelism receives threads; num_images includes the query itself.
    Spatial verification is disabled to preserve the existing retrieval protocol.
    """
    if min(threads, topk, repeats) < 1:
        raise ValueError("BoW threads, topk and repeats must be positive")
    return [
        str(executable),
        "vocab_tree_retriever",
        *_command_options(
            {
                "database_path": database,
                "num_threads": threads,
                "num_images": topk + 1,
                "num_images_after_verification": 0,
                "log_target": "stderr",
                "prepared_path": prepared,
                "repeats": repeats,
            }
        ),
    ]


def _command_options(options):
    return [
        part for name, value in options.items() for part in ("--" + name, str(value))
    ]


def build_prepared_colmap(
    colmap, *, adapter_sources, source_root=None, build_root="build/colmap_bow"
):
    """Build the prepared-BoW adapter without modifying the COLMAP installation.

    ``adapter_sources`` contains colmap_prepared.patch and prepared_bow.h.
    ``source_root`` is a matching COLMAP checkout with build-vcpkg development
    artifacts. Compilation occurs only when this function is called explicitly;
    outputs live beneath build_root, outside the static-result cache.
    """
    source = Path(source_root or Path(colmap).resolve().parents[2]).resolve()
    build = source / "build-vcpkg"
    native = Path(adapter_sources).resolve()
    patch = native / "colmap_prepared.patch"
    helper = native / "prepared_bow.h"
    commands_path = build / "compile_commands.json"
    if not commands_path.is_file():
        raise FileNotFoundError(f"COLMAP development build required: {commands_path}")
    commands = json.loads(commands_path.read_text())
    fingerprint = hashlib.sha256(patch.read_bytes() + helper.read_bytes())
    paths = [
        line[6:] for line in patch.read_text().splitlines() if line.startswith("--- a/")
    ]
    for relative in paths:
        fingerprint.update((source / "src" / relative).read_bytes())
    fingerprint.update(commands_path.read_bytes())
    output = Path(build_root).resolve() / fingerprint.hexdigest()[:16]
    binary = output / "colmap"
    if binary.is_file():
        return binary.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied = output / "src"
    for relative in paths:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "src" / relative, target)
    print(f"Build isolated COLMAP adapter: {output}", flush=True)

    def build_command(command, directory):
        with (output / "build.log").open("a") as log:
            result = subprocess.run(
                command, cwd=directory, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(
                f"COLMAP adapter build failed; see {output / 'build.log'}"
            )

    build_command(["patch", "-p1", "-i", str(patch.resolve())], copied)
    objects = {}
    for suffix in ("/exe/vocab_tree.cc", "/retrieval/visual_index.cc"):
        entry = next(c for c in commands if c["file"].endswith(suffix))
        command = shlex.split(entry["command"])
        obj = (output / (Path(entry["file"]).stem + ".o")).resolve()
        command[command.index("-o") + 1] = str(obj)
        command = [
            str((copied / Path(c).relative_to(source / "src")).resolve())
            if c == entry["file"]
            else c
            for c in command
        ]
        command[1:1] = ["-I" + str(copied.resolve()), "-I" + str(native.resolve())]
        build_command(command, build)
        objects[Path(entry["file"]).name] = str(obj)
    link = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", "src/colmap/exe/colmap"],
        text=True,
    ).splitlines()[-1]
    if not link.startswith(": && ") or not link.endswith(" && :"):
        raise ValueError("Unsupported COLMAP link command")
    command = shlex.split(link[5:-5])
    command[command.index("-o") + 1] = str(binary.with_suffix(".partial").resolve())
    position = next(
        i
        for i, c in enumerate(command)
        if c.endswith("/colmap_main.dir/vocab_tree.cc.o")
    )
    command[position] = objects["vocab_tree.cc"]
    command.insert(position + 1, objects["visual_index.cc"])
    build_command(command, build)
    binary.with_suffix(".partial").replace(binary)
    return binary.resolve()


def run_colmap(command, log, *, environment=None, monitor=None, append=False):
    """Stream a native command and query progress to a caller-owned run directory.

    ``monitor`` is an optional context manager supplied by the caller. Backend
    execution does not depend on a benchmark package, resource sampler or CLI.
    Return process wall seconds; nonzero exits include the log path in the error.
    """
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    child_environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    child_environment.update(environment or {})
    last_update = 0
    with (
        log.open("a" if append else "w", buffering=1) as stream,
        monitor if monitor is not None else nullcontext(),
    ):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=child_environment,
            text=True,
            errors="replace",
        )
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                match = re.search(r"Retrieval trial (\d+): (\d+)/(\d+)", line)
                if match:
                    trial, done, total = map(int, match.groups())
                    atomic_json(
                        log.with_name("progress.json"),
                        dict(
                            stage="query",
                            trial=trial,
                            completed_queries=done,
                            total_queries=total,
                            updated_at_unix=time.time(),
                        ),
                    )
                if (
                    "Encoded " in line
                    or "Retrieval trial " in line
                    or "Pure retrieval" in line
                ):
                    if time.monotonic() - last_update >= 10 or "Pure retrieval" in line:
                        print(line.strip(), flush=True)
                        last_update = time.monotonic()
            code = process.wait()
        finally:
            # Only this call's child is owned here. A progress/logging failure
            # must not leave a native stage running after its caller has failed.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
    if code:
        raise RuntimeError(f"COLMAP exited {code}; see {log}")
    return time.perf_counter() - started


def parse_bow_log(path):
    """Read legacy COLMAP query durations and directed non-self pairs."""
    path = Path(path)
    query_times = []
    pairs = []
    query = None
    for line in path.read_text().splitlines():
        match = re.search(r"Querying for image (.+) \[\d+/\d+\]", line)
        if match:
            query = match[1]
        elif query is not None and (match := re.search(r" in ([\d.]+)s", line)):
            query_times.append(float(match[1]))
        elif query is not None and (
            match := re.search(r"image_id=\d+, image_name=(.+), score=", line)
        ):
            if match[1] != query:
                pairs.append((query, match[1]))
    return query_times, pairs
