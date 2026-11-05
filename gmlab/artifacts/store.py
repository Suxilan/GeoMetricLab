"""Local POSIX artifact publication, verification and references; no compute IO."""

import json
import os
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..utils.io import atomic_json
from .identity import canonical_json, file_digest, identity_key


def component(value):
    if not isinstance(value, str) or not re.fullmatch("[a-z0-9][a-z0-9_-]*", value):
        raise ValueError(f"Invalid artifact path component: {value!r}")
    return value


def object_file(root, name):
    """Constrain manifest paths to ordinary files inside this object."""
    relative = Path(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in (".", "..") for p in relative.parts)
    ):
        raise ValueError("Payload paths must stay inside the object")
    path = root / relative
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("Artifact payloads cannot traverse symlinks")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Payload escaped object")
    return path


class ArtifactStore:
    """One local filesystem store. Explicit root > env > local cache default; no implicit version directory.

    Identity requires schema version, kind/source/selection and calculation.
    In schema 2 selection is a browsing method namespace, excluded from the key.
    Display recipe and usage are external to identity. Only completed objects
    participate in lookup. Advisory flock coordinates cooperating POSIX writers;
    network filesystems and hostile concurrent filesystem mutation are untested.
    """

    def __init__(self, root=None):
        default = Path("cache")
        self.root = (
            Path(root or os.environ.get("GMLAB_CACHE_ROOT", default))
            .expanduser()
            .resolve()
        )

    def _scope(self, identity):
        if (
            identity.get("schema_version") not in (1, 2, 3, 4)
            or "calculation" not in identity
        ):
            raise ValueError("Expected identity schema_version=1/2/3/4 and calculation")
        return Path(
            *(component(identity[name]) for name in ("kind", "source", "selection"))
        )

    @staticmethod
    def _contract(identity):
        # Schema 2 separates a browsing method name from the calculation key.
        # Schema 1 remains readable for already written development artifacts.
        if identity.get("schema_version") in (2, 3, 4):
            return {
                name: value for name, value in identity.items() if name != "selection"
            }
        return identity

    @classmethod
    def key(cls, identity):
        return identity_key(cls._contract(identity))

    @classmethod
    def _same_calculation(cls, saved, requested):
        return canonical_json(cls._contract(saved)) == canonical_json(
            cls._contract(requested)
        )

    def verify(self, path, *, identity=None, full=True):
        """Validate complete manifest and payloads; full=False checks sizes only.

        Full validation streams SHA256; fast opens never claim content checking.
        This verifies declared identity/integrity, not correctness of computation.
        """
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        if (
            manifest.get("schema_version") != 1
            or manifest.get("state") != "complete"
            or manifest.get("provenance_status") != "verified"
        ):
            raise ValueError("Object is not a verified completed artifact")
        actual = manifest["identity"]
        self._scope(actual)
        if manifest["artifact_key"] != self.key(actual):
            raise ValueError("Manifest identity digest mismatch")
        if identity is not None and not self._same_calculation(actual, identity):
            raise ValueError("Full calculation identity mismatch")
        if not manifest.get("payloads"):
            raise ValueError("No payloads")
        for payload in manifest["payloads"]:
            target = object_file(path, payload["path"])
            if not target.is_file() or target.stat().st_size != payload["bytes"]:
                raise ValueError("Missing/truncated payload")
            if full and file_digest(target) != payload["sha256"]:
                raise ValueError("Payload checksum mismatch")
        return manifest

    def lookup(self, identity, *, full=True):
        scope, key = self._scope(identity), self.key(identity)
        parent = self.root / scope
        parents = [parent]
        if identity.get("schema_version") in (2, 3, 4) and parent.parent.exists():
            parents.extend(
                p
                for p in sorted(parent.parent.iterdir())
                if p != parent and p.is_dir() and not p.is_symlink()
            )
        for candidate in parents:
            for name in (key[:16], key):
                path = candidate / name
                if path.is_dir():
                    manifest = json.loads((path / "manifest.json").read_text())
                    if manifest.get("artifact_key") == key:
                        self.verify(path, identity=identity, full=full)
                        return path
        return None

    @contextmanager
    def write(self, identity, recipe=None, *, resume=True):
        """Hold full-key lock; recheck hit, otherwise yield a private staging tx.

        Exceptions retain staging for explicit same-identity resume. Readers never
        see it. Format writers own safe flush/close and resume boundaries. Commit
        must happen inside this context; published payloads cannot be appended.
        """
        import fcntl  # Local POSIX locking is needed only by the writer path.

        if recipe is not None:
            component(recipe)
        self._scope(identity)
        identity = json.loads(canonical_json(identity))
        key = self.key(identity)
        lock = self.root / ".locks" / f"{key}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            tx = None
            try:
                hit = self.lookup(identity)
                stage = None
                if hit is None:
                    staging = self.root / ".staging"
                    staging.mkdir(exist_ok=True)
                    if resume:
                        for candidate in sorted(staging.iterdir()):
                            request = candidate / "request.json"
                            if request.is_file() and self._contract(
                                json.loads(request.read_text()).get("identity", {})
                            ) == self._contract(identity):
                                stage = candidate
                                break
                    if stage is None:
                        stage = staging / uuid.uuid4().hex
                        stage.mkdir()
                        atomic_json(stage / "request.json", {"identity": identity})
                tx = {
                    "path": hit or stage,
                    "hit": hit is not None,
                    "identity": identity,
                    "recipe": recipe,
                    "_active": True,
                }
                yield tx
            finally:
                if tx is not None:
                    tx["_active"] = False
                fcntl.flock(stream, fcntl.LOCK_UN)

    @contextmanager
    def export(self, identity, output, *, overwrite=False):
        """Use the same staged payload/manifest contract for a caller-owned H5.

        Exports can share a directory with run records. The exclusive lock covers
        all writes; the manifest is replaced last. Shared readers must verify it.
        Cache objects are immutable; overwrite applies only to explicit exports.
        """
        import fcntl

        output = Path(output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self._scope(identity)
        key = self.key(identity)
        # Sidecars belong to one feature export per directory.
        lock_path = output.parent / ".export.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            tx = None
            try:
                existing = output.parent / "manifest.json"
                hit = False
                if existing.exists():
                    saved = json.loads(existing.read_text())
                    if saved.get("artifact_key") == key:
                        try:
                            self.verify(output.parent, identity=identity)
                            hit = any(
                                p["path"] == output.name for p in saved["payloads"]
                            )
                        except ValueError:
                            if not overwrite:
                                raise
                    if not hit and not overwrite:
                        raise FileExistsError(
                            "Export identity changed; choose another destination or explicit overwrite"
                        )
                elif output.exists() and not overwrite:
                    raise FileExistsError(
                        "Unverified export exists; explicit overwrite required"
                    )
                staging = output.parent / ".staging"
                stage = output.parent if hit else None
                if not hit:
                    if staging.exists():
                        for candidate in sorted(staging.iterdir()):
                            request = candidate / "request.json"
                            if request.is_file():
                                saved_request = json.loads(request.read_text())
                                if saved_request.get(
                                    "output"
                                ) == output.name and self._same_calculation(
                                    saved_request.get("identity", {}), identity
                                ):
                                    stage = candidate
                                    break
                    if stage is None:
                        stage = staging / uuid.uuid4().hex
                        stage.mkdir(parents=True)
                        atomic_json(
                            stage / "request.json",
                            {"identity": identity, "output": output.name},
                        )
                tx = {
                    "path": stage,
                    "output": output,
                    "filename": output.name,
                    "hit": hit,
                    "identity": identity,
                    "_active": True,
                }
                yield tx
            finally:
                if tx is not None:
                    tx["_active"] = False
                fcntl.flock(lock, fcntl.LOCK_UN)

    def commit(self, tx, payloads, *, metadata=None, parents=()):
        """Publish closed, format-validated files; payloads declare path/schema.

        Files get complete byte hashes, then one directory rename publishes them.
        Read-only permissions discourage accidental edits; they are not a security
        boundary against the owner. No overwrite/recompute or deletion API exists.
        """
        if not tx.get("_active") or tx["hit"]:
            raise ValueError("Commit requires an active staging transaction")
        stage, identity = tx["path"], tx["identity"]
        if not tx.get("output") and stage.parent != self.root / ".staging":
            raise ValueError("Invalid staging directory")
        records = []
        for payload in payloads:
            target = object_file(stage, payload["path"])
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            records.append(
                {
                    **payload,
                    "bytes": target.stat().st_size,
                    "sha256": file_digest(target),
                }
            )
        key = self.key(identity)
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kind": identity["kind"],
            "artifact_key": key,
            "identity": identity,
            "state": "complete",
            "provenance_status": "verified",
            "payloads": records,
            "metadata": metadata or {},
            "parents": list(parents),
        }
        atomic_json(stage / "manifest.json", manifest)
        self.verify(stage, identity=identity)
        if tx.get("output"):
            destination = tx["output"].parent
            # Manifest is the commit marker. Readers use the same export lock;
            # an interrupted replacement never passes checksum validation.
            for payload in records:
                os.replace(stage / payload["path"], destination / payload["path"])
            os.replace(stage / "manifest.json", destination / "manifest.json")
            (stage / "request.json").unlink(missing_ok=True)
            stage.rmdir()
            tx.update(path=destination, hit=True)
            return destination
        parent = self.root / self._scope(identity)
        parent.mkdir(parents=True, exist_ok=True)
        destination = parent / key[:16]
        if destination.exists():
            destination = parent / key
        if destination.exists():
            raise FileExistsError(destination)
        (stage / "request.json").unlink(missing_ok=True)
        try:
            os.rename(stage, destination)
        except BaseException:
            atomic_json(stage / "request.json", {"identity": identity})
            raise
        for file in destination.rglob("*"):
            if file.is_file():
                file.chmod(0o444)
            elif file.is_dir():
                file.chmod(0o555)
        destination.chmod(0o555)
        for directory in (destination, parent):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        tx.update(path=destination, hit=True)
        return destination

    def reference(self, path):
        manifest = self.verify(path)
        relative = Path(path).relative_to(self.root)
        return {
            "artifact_key": manifest["artifact_key"],
            "object": str(relative),
            "payloads": manifest["payloads"],
        }

    def resolve(self, kind, source, method, *, artifact_key=None, full=True):
        """Locate one published result for explicit data/method selection.

        Multiple calculation variants require a full key; never pick the newest
        file or infer a hash from a legacy path. Missing results do not run code.
        """
        kind, source, method = (component(x) for x in (kind, source, method))
        candidates = []
        for path in (self.root / kind / source / method).glob("*/manifest.json"):
            saved = json.loads(path.read_text())
            identity = saved.get("identity", {})
            if (
                saved.get("state") != "complete"
                or saved.get("provenance_status") != "verified"
                or identity.get("kind") != kind
                or identity.get("source") != source
                or identity.get("selection") != method
            ):
                continue
            if artifact_key is None or saved.get("artifact_key") == artifact_key:
                candidates.append(path.parent)
        if not candidates:
            raise FileNotFoundError(f"No published result: {kind}/{source}/{method}")
        if len(candidates) != 1:
            raise ValueError(
                "Multiple calculation variants; specify a full artifact_key"
            )
        self.verify(candidates[0], full=full)
        return candidates[0]

    def inventory(self):
        """List completed namespace manifests, excluding private writer areas."""
        return [
            {"path": str(p.parent), **json.loads(p.read_text())}
            for p in sorted(self.root.glob("*/*/*/*/manifest.json"))
            if not p.relative_to(self.root).parts[0].startswith(".")
        ]
