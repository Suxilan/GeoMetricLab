"""Chunked finite global-feature H5 IO; ordered sample rows are never deduplicated."""

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal
from contextlib import contextmanager, nullcontext
from pathlib import Path

import h5py
import numpy as np


def set_feature_source_verified(path: str | Path, *, verified: bool) -> None:
    """Record source verification in closed private staging under its tx lock.

    The publisher must hold the transaction lock. Never pass a published file.
    Mark True only after the producer validates all ranks' sources. False also
    clears completion marks; the pending flag written before production prevents
    reuse even when this invalidation IO fails. No writer object is closed here.
    """
    if type(verified) is not bool:
        raise TypeError("verified must be boolean")
    path = Path(path)
    if not path.exists():
        return
    with h5py.File(path, "r+") as file:
        file.attrs["_source_verified"] = verified
        file.flush()
        if not verified and "written" in file:
            file["written"][:] = False
            file.flush()


class GlobalFeatureWriter:
    """Write finite [B,D] or fixed [B,C,H,W] rows into private staging H5.

    written[N] is flushed after features; resume is supported after clean close
    (including handled exceptions), not arbitrary SIGKILL/power-loss HDF5 repair.
    Completed rows are retained and must reproduce exactly if revisited. Duplicate
    physical samples use different row indices. No published file may be opened
    with this writer; the caller owns staging/locking and publication.
    The extraction producer additionally uses begin_source_validation before
    copying/predicting and certifies closed partial rows before allowing resume.
    """

    def __init__(
        self,
        path: str | Path,
        samples: Sequence[Mapping[str, Any]],
        *,
        dtype: str = "float32",
        shape: Sequence[int] | None = None,
    ) -> None:
        self.dtype = np.dtype(dtype)
        if self.dtype not in (
            np.dtype("float16"),
            np.dtype("float32"),
            np.dtype("float64"),
        ):
            raise ValueError("Global feature storage supports float16/32/64")
        self.row_shape = None if shape is None else tuple(shape)
        if self.row_shape is not None and (
            len(self.row_shape) not in (1, 3)
            or any(type(d) is not int or d < 1 for d in self.row_shape)
        ):
            raise ValueError("Feature rows require fixed [D] or [C,H,W] shape")
        self.count = len(samples)
        if not self.count:
            raise ValueError("Cannot publish empty feature sets")
        self.file = h5py.File(path, "a")
        try:
            names = [sample["sample_id"] for sample in samples]
            if "written" not in self.file:
                self.file.create_dataset(
                    "written", data=np.zeros(self.count, dtype=bool)
                )
                self.file.create_dataset(
                    "image_names", data=names, dtype=h5py.string_dtype("utf-8")
                )
            if (
                self.file["written"].shape != (self.count,)
                or self.file["image_names"].asstr()[:].tolist() != names
            ):
                raise ValueError("Staging sample order changed")
            if "features" in self.file:
                if self.file["features"].dtype != self.dtype:
                    raise ValueError("Staging storage dtype changed")
                if (
                    self.row_shape is not None
                    and self.file["features"].shape[1:] != self.row_shape
                ):
                    raise ValueError("Staging feature layout changed")
        except BaseException:
            self.file.close()
            raise

    def begin_source_validation(self) -> None:
        """Require source verification before any newly produced rows can resume.

        Existing certified partial rows remain reusable. Unverified or historical
        staging is conservatively recomputed. Flush the pending flag before any
        copy/inference; failure to record it must prevent new production.
        """
        recorded = self.file.attrs.get("_source_verified", False)
        resumable = isinstance(recorded, (bool, np.bool_)) and bool(recorded)
        self.file.attrs["_source_verified"] = False
        self.file.flush()
        if not resumable:
            self.file["written"][:] = False
            self.file.flush()

    def write(self, indices: Sequence[int] | np.ndarray, embeddings: Any) -> None:
        """Write aligned rows, then flush their written mask.

        indices is unique integer [B] in [0,N). embeddings is a NumPy-compatible
        array or Tensor with finite [B,D]/[B,C,H,W] values. Tensors are detached
        and copied to CPU; conversion to storage dtype is explicit. Revisited
        completed rows must match exactly. The caller keeps the lock and must
        close this writer before certification/publication, including on errors.
        """
        raw_indices = np.asarray(indices)
        if raw_indices.dtype.kind not in ("i", "u"):
            raise TypeError("Stable row indices must be integers")
        if raw_indices.ndim != 1:
            raise ValueError("Stable row indices must be one-dimensional")
        indices = raw_indices.astype(np.int64, copy=False)
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu()
            if str(embeddings.dtype) == "torch.bfloat16":
                embeddings = embeddings.float()
            embeddings = embeddings.numpy()
        values = np.asarray(embeddings, dtype=self.dtype)
        if (
            values.ndim not in (2, 4)
            or len(values) != len(indices)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Expected finite aligned [batch,D] features")
        if (
            indices.ndim != 1
            or len(set(indices.tolist())) != len(indices)
            or (indices < 0).any()
            or (indices >= self.count).any()
        ):
            raise ValueError("Invalid/duplicate stable row indices")
        if self.row_shape is not None and values.shape[1:] != self.row_shape:
            raise ValueError("Feature layout differs from the declared stream")
        if "features" not in self.file:
            row_bytes = max(int(np.prod(values.shape[1:])) * self.dtype.itemsize, 1)
            chunk_rows = min(self.count, 256, max(1, (4 * 1024 * 1024) // row_bytes))
            self.file.create_dataset(
                "features",
                shape=(self.count, *values.shape[1:]),
                dtype=self.dtype,
                chunks=(chunk_rows, *values.shape[1:]),
            )
        features = self.file["features"]
        if features.shape != (self.count, *values.shape[1:]):
            raise ValueError("Embedding shape changed across batches")
        order = np.argsort(indices)
        indices, values = indices[order], values[order]
        written = self.file["written"][indices]
        if written.any() and not np.array_equal(
            features[indices[written]], values[written]
        ):
            raise ValueError("Resumed completed rows differ; use a new realization")
        new = ~written
        if new.any():
            features[indices[new]] = values[new]
            self.file.flush()
            self.file["written"][indices[new]] = True
            self.file.flush()

    def close(self) -> None:
        """Close the owned H5 handle; transaction locks remain caller-owned."""
        self.file.close()


def validate_global_features(
    path: str | Path,
    count: int,
    *,
    chunk_rows: int = 256,
) -> dict[str, Any]:
    """Validate a closed [N,D] writer payload through the shared stream schema."""
    schema = validate_feature_stream(path, count, chunk_rows=chunk_rows)
    if len(schema["shape"]) != 2:
        raise ValueError("Invalid global feature shape")
    return schema


def iter_global_features(
    path: str | Path | None = None,
    *,
    record: Mapping[str, Any] | str | Path | None = None,
    stream: str | None = None,
    chunk_rows: int = 256,
    cache_root: str | Path | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield global/spatial row chunks under shared verification and locking.

    Arguments and copied (start, rows) results follow iter_feature_stream.
    Unlike that strict reader, bare paths also accept old unverified files;
    callers requiring identity/checksum binding must pass a record. Despite the
    historical name, spatial rows are also accepted. Exhaust or close the iterator
    on early exit so the shared H5 handle/read lock is released promptly.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    with open_feature_stream(
        path,
        record=record,
        stream=stream,
        cache_root=cache_root,
        allow_unverified=record is None,
        chunk_rows=chunk_rows,
    ) as (_, values):
        for start in range(0, len(values), chunk_rows):
            yield start, values[start : start + chunk_rows]


def write_samples(
    path: str | Path,
    samples: Sequence[Mapping[str, Any]],
) -> None:
    """Persist ordered row identities plus deployment locators, including repeats."""
    with Path(path).open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(samples):
            if sample["row_id"] != index:
                raise ValueError("Sample row_id must match stable index")
            stream.write(json.dumps(sample, sort_keys=True, allow_nan=False) + "\n")


def _resolve_feature_input(path, record, cache_root, stream=None):
    """Resolve one record/path, honoring explicit and environment cache roots."""
    from ...artifacts import ArtifactStore
    from ...artifacts.store import object_file

    if (path is None) == (record is None):
        raise ValueError("Choose feature path or extraction record")
    run = None
    if record is not None:
        if isinstance(record, Mapping):
            run, record_path = dict(record), Path.cwd() / "record.json"
        else:
            record_path = Path(record).resolve()
            run = json.loads(record_path.read_text())
        if "streams" in run:
            if stream is None:
                raise ValueError("Multi-stream record requires an explicit stream")
            if stream not in run["streams"]:
                raise ValueError(f"Unknown feature stream: {stream}")
            run = dict(run["streams"][stream])
        path = Path(run["features"])
        if not path.is_absolute():
            path = record_path.parent / path
    if record is None and stream is not None:
        raise ValueError("stream selection requires an extraction record")
    path = Path(path).resolve()
    reference = {} if run is None else run.get("reference", {})
    preferred_root = (
        cache_root if cache_root is not None else os.environ.get("GMLAB_CACHE_ROOT")
    )
    if (
        reference.get("object")
        and reference.get("feature_payload")
        and (cache_root is not None or not path.is_file())
    ):
        directory = object_file(ArtifactStore(preferred_root).root, reference["object"])
        path = object_file(directory, reference["feature_payload"])
    return path, run


@contextmanager
def _feature_read_lock(path):
    """Hold one shared export lock until the caller has closed its H5 handle."""
    import fcntl

    lock_path = path.parent / ".export.lock"
    with lock_path.open("rb") if lock_path.exists() else nullcontext(None) as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            yield
        finally:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _feature_file_schema(
    file, path, *, count=None, shape=None, require_written=False, chunk_rows=256
):
    """Check an already open global/spatial dataset with bounded validation IO."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    values = file["features"]
    actual = values.shape
    if (
        len(actual) not in (2, 4)
        or values.dtype.kind != "f"
        or any(d < 1 for d in actual[1:])
    ):
        raise ValueError("Invalid feature stream layout/dtype")
    if count is not None and actual[0] != count:
        raise ValueError("Invalid feature row count")
    count = actual[0]
    if shape is not None and tuple(actual[1:]) != tuple(shape):
        raise ValueError("Feature stream differs from declared shape")
    if file["image_names"].shape != (count,):
        raise ValueError("Invalid feature row mapping")
    if require_written and "written" not in file:
        raise ValueError("Incomplete feature stream: missing written-row mask")
    if "written" in file and file["written"].shape != (count,):
        raise ValueError("Invalid written-row mapping")
    if int(file.attrs.get("committed_rows", count)) != count:
        raise ValueError("Uncommitted rows in legacy feature file")
    row_bytes = max(int(np.prod(actual[1:])) * values.dtype.itemsize, 1)
    validation_rows = min(chunk_rows, max(1, (4 * 1024 * 1024) // row_bytes))
    for start in range(0, count, validation_rows):
        stop = start + validation_rows
        if "written" in file and not file["written"][start:stop].all():
            raise ValueError("Incomplete features")
        if not np.isfinite(values[start:stop]).all():
            raise ValueError("Nonfinite features")
    return {
        "path": path.name,
        "format": "gmlab_global_h5_v1" if len(actual) == 2 else "gmlab_spatial_h5_v1",
        "shape": list(actual),
        "dtype": str(values.dtype),
    }


def _verified_feature_manifest(path, run, allow_unverified):
    """Bind the requested record to the exact payload under the caller's lock."""
    from ...artifacts import ArtifactStore

    if not (path.parent / "manifest.json").exists():
        if allow_unverified and run is None:
            return None, None
        raise ValueError("A verified manifest is required")
    manifest = ArtifactStore().verify(path.parent)
    payload = next(
        (
            item
            for item in manifest["payloads"]
            if item["path"] == path.name
            and item.get("format") in {"gmlab_global_h5_v1", "gmlab_spatial_h5_v1"}
        ),
        None,
    )
    if payload is None:
        raise ValueError("Requested feature file is not a declared payload")
    if run is not None and (
        run["artifact_key"] != manifest["artifact_key"]
        or run["features_sha256"] != payload["sha256"]
    ):
        raise ValueError("Extraction record identity/checksum mismatch")
    return manifest, payload


@contextmanager
def open_feature_stream(
    path: str | Path | None = None,
    *,
    record: Mapping[str, Any] | str | Path | None = None,
    stream: str | None = None,
    cache_root: str | Path | None = None,
    allow_unverified: bool = False,
    expected_layout: Literal["global"] | None = None,
    expected_names: Sequence[str] | None = None,
    expected_root: str | Path | None = None,
    chunk_rows: int = 256,
) -> Iterator[tuple[dict[str, Any], h5py.Dataset]]:
    """Resolve, verify and yield (metadata, sliceable H5 dataset) for one stream.

    Choose exactly one of path or record. record is an extraction-record mapping
    or JSON path; a bundle requires stream. Relative feature paths are resolved
    from the record file (cwd for mappings). cache_root overrides the environment
    root for a portable cache reference. Path-only legacy files require explicit
    allow_unverified=True; records always require verified manifests/checksums.
    expected_layout="global" requires [N,D]; None also accepts fixed [N,C,H,W].
    expected_names checks exact ordered names; with expected_root it checks each
    recorded locator against root/name instead. Repeated physical inputs remain
    distinct stable row positions; rows are never sorted or deduplicated here.

    metadata contains features=None, path (resolved H5 string), shape (list), dtype
    (NumPy name), image_names (ordered sample IDs), manifest (dict or None),
    verified (bool), metadata (H5 attrs as JSON-like values), and record (resolved
    identity/checksum mapping or None for unverified legacy files). All payload
    hashes, schema, finite values and stable-row mappings are checked before yield;
    chunk_rows bounds validation buffers, not total IO or metadata memory.

    The shared export lock, if present, and single read-only H5 handle remain held
    for the entire context, including exception exit. Do not retain the H5 dataset
    after exit; copied NumPy slices and metadata may outlive it. No writer/staging
    is opened. Malformed or mismatched contracts raise ValueError; missing files,
    malformed JSON and H5 errors propagate. There is no reader recovery/repair.
    """
    path, run = _resolve_feature_input(path, record, cache_root, stream)
    from .feature_description import validate_feature_output

    with _feature_read_lock(path):
        manifest, payload = _verified_feature_manifest(path, run, allow_unverified)
        if stream is not None:
            output = manifest["identity"]["calculation"].get("output", {})
            if output.get("stream", "global") != stream:
                raise ValueError(f"Unknown feature stream: {stream}")
        with h5py.File(path, "r") as file:
            schema = _feature_file_schema(
                file, path, require_written=manifest is not None, chunk_rows=chunk_rows
            )
            if expected_layout == "global" and len(schema["shape"]) != 2:
                raise ValueError("Expected global features [N,D]")
            if expected_layout not in (None, "global"):
                raise ValueError("Unknown requested feature layout")
            names = file["image_names"].asstr()[:].tolist()
            samples = None
            bound_record = None
            if manifest is not None:
                validate_feature_output(
                    manifest["identity"]["calculation"].get("output", {}),
                    schema["shape"][1:],
                )
                if any(
                    schema[key] != payload[key] for key in ("format", "shape", "dtype")
                ):
                    raise ValueError("Feature schema differs from manifest")
                samples = [
                    json.loads(line)
                    for line in (path.parent / "samples.jsonl").read_text().splitlines()
                ]
                if len(samples) != len(names) or any(
                    item["row_id"] != i or item["sample_id"] != names[i]
                    for i, item in enumerate(samples)
                ):
                    raise ValueError("Sample row mapping differs from feature names")
                bound_record = {
                    **(run or {}),
                    "features": str(path),
                    "artifact_key": manifest["artifact_key"],
                    "features_sha256": payload["sha256"],
                }
            if expected_names is not None:
                if expected_root is None:
                    if names != list(expected_names):
                        raise ValueError("Feature image names/order differ")
                else:
                    if samples is None:
                        raise ValueError(
                            "Root-aware alignment requires verified samples"
                        )
                    expected = [
                        (Path(expected_root) / name).resolve()
                        for name in expected_names
                    ]
                    if expected != [
                        Path(item["locator"]).resolve() for item in samples
                    ]:
                        raise ValueError("Feature image paths/order differ")
            metadata = {
                "features": None,
                "path": str(path),
                "shape": schema["shape"],
                "dtype": schema["dtype"],
                "image_names": names,
                "manifest": manifest,
                "verified": manifest is not None,
                "metadata": {
                    key: np.asarray(value).tolist() for key, value in file.attrs.items()
                },
                "record": bound_record,
            }
            yield metadata, file["features"]


def read_global_features(
    path: str | Path | None = None,
    *,
    record: Mapping[str, Any] | str | Path | None = None,
    stream: str | None = None,
    expected_names: Sequence[str] | None = None,
    expected_root: str | Path | None = None,
    allow_unverified: bool = False,
    load_values: bool = True,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return detached metadata and optional finite global features [N,D].

    Arguments, identity checks, name alignment and legacy opt-in follow
    open_feature_stream, with layout restricted to global. The returned dict has
    its documented metadata fields and features=np.ndarray when load_values=True,
    otherwise None. Shape/dtype describe stored values; they do not imply forward
    precision. Loading allocates the full matrix; False still verifies all rows
    and payloads. The read lock/H5 handle close before return, including errors.
    """
    with open_feature_stream(
        path,
        record=record,
        stream=stream,
        cache_root=cache_root,
        allow_unverified=allow_unverified,
        expected_layout="global",
        expected_names=expected_names,
        expected_root=expected_root,
    ) as (data, values):
        return {**data, "features": values[:] if load_values else None}


@contextmanager
def open_global_features(
    path: str | Path | None = None,
    *,
    record: Mapping[str, Any] | str | Path | None = None,
    stream: str | None = None,
    cache_root: str | Path | None = None,
) -> Iterator[tuple[dict[str, Any], h5py.Dataset]]:
    """Yield (metadata, H5 [N,D] dataset) under open_feature_stream's contract.

    Accept either path or record, never both; bundles require stream. Verified
    manifests are mandatory. The dataset is valid only inside the with block;
    metadata follows open_feature_stream. Exit always releases handle/read lock.
    """
    with open_feature_stream(
        path,
        record=record,
        stream=stream,
        cache_root=cache_root,
        expected_layout="global",
    ) as opened:
        yield opened


def read_feature_stream(
    record: Mapping[str, Any] | str | Path,
    *,
    stream: str | None = None,
    load_values: bool = True,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return verified global/spatial metadata with optional NumPy features.

    record is a mapping or JSON path; bundles require stream. Returned fields and
    relocation follow open_feature_stream. features is [N,D] or [N,C,H,W] when
    load_values=True, otherwise None (full verification still runs). Arrays own
    their data and all H5 handles/read locks close before return or exception.
    """
    with open_feature_stream(record=record, stream=stream, cache_root=cache_root) as (
        data,
        values,
    ):
        return {**data, "features": values[:] if load_values else None}


def iter_feature_stream(
    record: Mapping[str, Any] | str | Path,
    *,
    stream: str | None = None,
    chunk_rows: int = 256,
    cache_root: str | Path | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (start, NumPy rows) in stable global/spatial stored-row order.

    record/stream/cache_root use open_feature_stream's verified contract.
    Each copied array has at most chunk_rows rows and shape [K,D]/[K,C,H,W];
    start is its zero-based offset, not a sample ID. Initial verification scans
    the full payload. The read lock/H5 handle span iteration: exhaust the iterator
    or call close() on early exit (contextlib.closing is suitable). Yielded copies
    outlive closure. Positive chunk_rows is required; no staging is read/repaired.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    with open_feature_stream(
        record=record, stream=stream, cache_root=cache_root, chunk_rows=chunk_rows
    ) as (_, values):
        for start in range(0, len(values), chunk_rows):
            yield start, values[start : start + chunk_rows]


def feature_collection(
    *,
    paths: Sequence[str | Path] | None = None,
    records: Sequence[Mapping[str, Any] | str | Path] | None = None,
    stream: str | None = None,
    expected_output: Mapping[str, Any] | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate compatible global caches without retaining feature matrices/handles.

    Choose a nonempty sequence of paths or records, not both. Bundles require
    stream. expected_output optionally constrains output semantic fields;
    cache_root supports relocated records. All inputs must have verified state
    identities and compatible numerical feature spaces; data selections may differ.
    Duplicate artifact keys are rejected; weighting is the consumer's choice.

    Return inputs (read_global_features metadata with features=None), artifact_keys
    in selection order, calculation (shared feature-space signature), cache_root
    (str or None), and shape=[sum(N),D]. No lock/handle survives return; subsequent
    iter_feature_collection reopens and verifies each input. Malformed/incompatible
    inputs raise ValueError with their input index; filesystem errors propagate.
    """
    if (paths is None) == (records is None):
        raise ValueError("Choose feature paths or extraction records")
    sources = paths if paths is not None else records
    if (
        isinstance(sources, (str, bytes))
        or not isinstance(sources, Sequence)
        or not sources
    ):
        raise ValueError(
            "Feature inputs must be a nonempty sequence of paths or records"
        )
    inputs, keys, reference = [], [], None
    from .feature_description import feature_space, validate_feature_space

    for index, source in enumerate(sources):
        try:
            data = read_global_features(
                source if paths else None,
                record=source if records else None,
                stream=stream,
                load_values=False,
                cache_root=cache_root,
            )
            manifest = data["manifest"]
            calculation = manifest["identity"]["calculation"]
            if not calculation.get("state_sha256"):
                raise ValueError("Missing actual model state identity")
            output = calculation.get("output", {})
            output_differences = [
                f"calculation.output.{key}"
                for key, value in (expected_output or {}).items()
                if output.get(key) != value
            ]
            if output_differences:
                raise ValueError(
                    "Feature output does not meet consumer requirements: "
                    + ", ".join(output_differences)
                )
            signature = feature_space(calculation, manifest=manifest)
            if reference is not None:
                validate_feature_space(signature, reference)
            reference = signature
            key = manifest["artifact_key"]
            if key in keys:
                raise ValueError("Duplicate feature artifact")
            keys.append(key)
            inputs.append(data)
        except ValueError as error:
            raise ValueError(f"Feature input {index}: {error}") from error
    dimensions = {item["shape"][1] for item in inputs}
    if len(dimensions) != 1:
        raise ValueError("Feature dimensions differ")
    return {
        "inputs": inputs,
        "artifact_keys": keys,
        "calculation": reference,
        "cache_root": None if cache_root is None else str(cache_root),
        "shape": [sum(item["shape"][0] for item in inputs), dimensions.pop()],
    }


def iter_feature_collection(
    collection: Mapping[str, Any],
    *,
    chunk_rows: int = 256,
) -> Iterator[np.ndarray]:
    """Yield copied finite [K,D] arrays in collection-input and stable-row order.

    collection is feature_collection's return mapping. Each chunk has at most
    chunk_rows rows; chunks never cross input boundaries. Every input is reopened
    and verified under its own read lock. Exhaust or close the iterator to release
    the current handle/lock; yielded arrays remain valid. No full matrix is held.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    for item in collection["inputs"]:
        with open_global_features(
            record=item["record"], cache_root=collection.get("cache_root")
        ) as (_, values):
            for start in range(0, len(values), chunk_rows):
                yield values[start : start + chunk_rows]


def validate_feature_stream(
    path: str | Path,
    count: int,
    *,
    shape: Sequence[int] | None = None,
    chunk_rows: int = 256,
) -> dict[str, Any]:
    """Check a closed staging payload's finite values and complete stable rows.

    count is the expected N; optional shape is [D] or [C,H,W]. Return the payload
    descriptor {path: basename, format, shape: list[int], dtype: NumPy name} for
    manifest publication. Own and close one read-only H5 handle; the publisher
    retains its transaction lock. This validates schema/rows, not source identity
    or a published manifest/checksum. Invalid data raises ValueError, IO propagates.
    """
    path = Path(path)
    with h5py.File(path, "r") as file:
        return _feature_file_schema(
            file,
            path,
            count=count,
            shape=shape,
            require_written=True,
            chunk_rows=chunk_rows,
        )
