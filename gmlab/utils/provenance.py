"""Read-only package code and dependency versions without experiment semantics."""

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path


def code_version() -> dict:
    """Record Git identity and current package sources, including untracked code.

    Source hashes remain available outside a Git checkout. Git unavailability is
    explicit; it never prevents using a model or evaluating an existing weight.
    """
    package = Path(__file__).resolve().parents[1]
    record = {
        "commit": None,
        "dirty": None,
        "gmlab_sha256": {
            str(path.relative_to(package)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(package.rglob("*.py"))
        },
    }
    try:
        record["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=package.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        record["dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=package.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return record


def dependency_versions() -> dict[str, str | None]:
    """Read distribution metadata without importing optional runtime backends."""
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "timm",
        "pytorch-lightning",
        "pytorch-metric-learning",
        "swanlab",
        "numpy",
        "Pillow",
        "PyYAML",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions
