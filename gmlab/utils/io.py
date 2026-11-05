"""File checksums, atomic replacement and strict JSON encoding (stdlib only)."""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


def file_digest(path: str | Path) -> str:
    """SHA256 of file bytes with bounded memory, including large H5 payloads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    """Return UTF-8 JSON bytes; mapping order is irrelevant, row order is not.

    Callers supply fully resolved JSON values, not object reprs or locators.
    Non-finite numbers and non-string mapping keys are rejected.
    """

    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("Identity mapping keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    check(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def atomic_write(path: str | Path, contents: bytes) -> None:
    """Replace one file after flushing a unique temporary file in its directory.

    This provides atomic visibility on a local filesystem and cleans temporary
    files after handled failures. It does not own locks, manifests or completion
    state, and makes no power-loss or network-filesystem durability guarantee.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any) -> None:
    """Atomically replace a strict canonical JSON document, with a final newline."""
    atomic_write(path, canonical_json(value) + b"\n")
