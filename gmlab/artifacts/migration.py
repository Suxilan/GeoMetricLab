"""Same-filesystem archival imports: unified paths, explicitly unknown calculation."""

import os
import uuid
from pathlib import Path

from ..utils.io import atomic_json
from .identity import file_digest, identity_key
from .store import component


def import_unverified(
    store,
    files,
    *,
    kind,
    source,
    selection,
    recipe,
    writer_stopped,
    evidence,
    validate=None,
):
    """Move a closed bundle without changing bytes; leave file-level read aliases.

    Caller must establish the writer is stopped and group dependent files. This
    API refuses DB/WAL/lock files and cross-filesystem moves. Unknown calculation
    is NEVER promoted to a reusable key: artifact_key=None, reuse_eligible=False.
    A content object_id addresses this archive, not a claimed calculation.
    Validation receives {original_path: staged_path}; errors roll back the move.
    """
    if not writer_stopped or not evidence:
        raise ValueError("A stopped-writer check and its evidence are required")
    parts = [component(value) for value in (kind, source, selection, recipe)]
    files = [Path(path).absolute() for path in files]
    if not files or len({p.name for p in files}) != len(files):
        raise ValueError("Expected a nonempty flat closed bundle")
    if any(
        p.is_symlink()
        or p.suffix in (".db", ".sqlite", ".lock", ".db-wal", ".db-shm")
        or p.name.endswith(("-wal", "-shm"))
        for p in files
    ):
        raise ValueError("Symlinks, databases, WAL and locks require separate handling")
    initial = {
        p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) for p in files
    }
    records = [
        {"path": p.name, "bytes": p.stat().st_size, "sha256": file_digest(p)}
        for p in files
    ]
    object_id = identity_key({"archive_payloads": records})
    destination = store.root / Path(*parts[:3]) / object_id[:16]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"Compare existing manifest/content; never overwrite: {destination}"
        )
    if any(p.stat().st_dev != destination.parent.stat().st_dev for p in files):
        raise ValueError(
            "Cross-filesystem migration requires copy/verify, not this move API"
        )
    for path, before in initial.items():
        after = path.stat()
        if before != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("Source changed while hashing")
    stage = store.root / ".staging" / uuid.uuid4().hex
    stage.mkdir(parents=True)
    moved = {}
    published = False
    try:
        for path in files:
            target = stage / path.name
            os.rename(path, target)
            moved[path] = target
        for record in records:
            if file_digest(stage / record["path"]) != record["sha256"]:
                raise ValueError("Migration checksum changed")
        validation = validate(moved) if validate else {"bytes_verified": True}
        manifest = {
            "schema_version": 1,
            "state": "imported",
            "provenance_status": "unverified",
            "reuse_eligible": False,
            "artifact_key": None,
            "object_id": object_id,
            "kind": kind,
            "source": source,
            "selection": selection,
            "identity": {"calculation": "unknown"},
            "payloads": records,
            "source_paths": [str(p) for p in files],
            "writer_check": evidence,
            "unverified_label": recipe,
            "validation": validation,
            "missing": [
                "actual_weights_and_buffers",
                "full_preprocessing",
                "output_space",
                "input_content_identity",
                "producer_revision",
            ],
            "pin": True,
        }
        atomic_json(stage / "manifest.json", manifest)
        os.rename(stage, destination)
        published = True
        for path in files:
            target = destination / path.name
            # Links preserve logical paths used by UCA and recursive glob callers.
            path.symlink_to(os.path.relpath(target, path.parent))
            target.chmod(0o444)
        (destination / "manifest.json").chmod(0o444)
        destination.chmod(0o555)
        return destination
    except BaseException:
        for original, staged in reversed(list(moved.items())):
            current = destination / staged.name if published else staged
            if original.is_symlink():
                original.unlink()
            if not original.exists() and current.exists():
                os.rename(current, original)
        raise
