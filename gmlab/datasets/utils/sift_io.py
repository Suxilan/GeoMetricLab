"""Publish a documented COLMAP SIFT database as a reusable local-feature result."""

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from numbers import Integral

import numpy as np

from ...artifacts import file_digest
from ...utils.io import atomic_json


def read_database(path):
    """Open an existing COLMAP SQLite database read-only, including committed WAL.

    The caller must close the connection (a SQLite transaction context alone does
    not close it). No missing database is created and no journal mode is changed.
    """
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def read_sift_record(record, *, store):
    """Resolve a published SIFT record to its verified database Path.

    A missing original location can use the cache-relative reference emitted by
    ArtifactStore. No dataset name, method name or directory hash is inferred.
    Missing inputs raise; this reader never extracts or republishes a database.
    """
    from ...artifacts.store import object_file

    if isinstance(record, dict):
        saved, parent = record, Path.cwd()
    else:
        record_path = Path(record).resolve()
        saved, parent = json.loads(record_path.read_text()), record_path.parent
    database = Path(saved["database"])
    if not database.is_absolute():
        database = parent / database
    reference = saved.get("reference", {})
    if database.is_file():
        directory = database.parent
    elif reference.get("object"):
        directory = object_file(store.root, reference["object"])
        database = None
    else:
        raise FileNotFoundError(database)
    manifest = store.verify(directory)
    if saved["artifact_key"] != manifest["artifact_key"]:
        raise ValueError("SIFT record identity mismatch")
    if reference and (
        reference.get("artifact_key") != manifest["artifact_key"]
        or reference.get("payloads") != manifest["payloads"]
    ):
        raise ValueError("SIFT record reference differs from published payloads")
    candidates = [
        item
        for item in manifest["payloads"]
        if item.get("format") == "colmap_sift_sqlite"
        and (database is None or item["path"] == database.name)
    ]
    if len(candidates) != 1:
        raise ValueError("SIFT record must select one COLMAP database payload")
    return object_file(directory, candidates[0]["path"])


def descriptor_rows(connection):
    """Iterate (image_name, rows, cols, uint8_blob), ordered by image name."""
    return connection.execute(
        "SELECT i.name,d.rows,d.cols,d.data FROM images i "
        "JOIN descriptors d ON i.image_id=d.image_id ORDER BY i.name"
    )


def local_descriptors(rows, cols, blob):
    """Decode quantized COLMAP L1_ROOT SIFT as unit-L2 float32 [rows,128].

    Quantization changes norms; re-normalization preserves the existing SIFT-VLAD
    input convention. Zero descriptors remain zero. Empty images return [0,128];
    the numerical consumer decides whether it can handle an empty image.
    """
    if rows < 0 or cols != 128 or blob is None or len(blob) != rows * cols:
        raise ValueError("Invalid COLMAP uint8 SIFT descriptor layout")
    values = np.frombuffer(blob, np.uint8).reshape(rows, cols).astype(np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return values


def iter_sift_descriptors(connection):
    """Yield (image_name, normalized [Ni,128]) without loading the database."""
    for name, rows, cols, blob in descriptor_rows(connection):
        yield name, local_descriptors(rows, cols, blob)


def sift_image_names(connection, *, require_descriptors=True, require_keypoints=False):
    """Return sorted image names with explicit descriptor/keypoint requirements."""
    descriptors = (
        " JOIN descriptors d ON i.image_id=d.image_id" if require_descriptors else ""
    )
    keypoints = (
        " JOIN keypoints k ON i.image_id=k.image_id" if require_keypoints else ""
    )
    return [
        row[0]
        for row in connection.execute(
            "SELECT i.name FROM images i" + descriptors + keypoints + " ORDER BY i.name"
        )
    ]


def sift_database_summary(connection, expected_names):
    """Check extracted membership and summarize descriptor counts/camera sizes."""
    if sift_image_names(connection, require_keypoints=True) != list(expected_names):
        raise ValueError("SIFT database has missing or extra images/features")
    counts = [row[0] for row in connection.execute("SELECT rows FROM descriptors")]
    cameras = connection.execute("SELECT width,height FROM cameras").fetchall()
    return {
        "descriptor_counts": {
            "min": min(counts),
            "median": float(np.median(counts)),
            "max": max(counts),
            "total": sum(counts),
        },
        "camera_sizes": sorted(set(cameras)),
    }


def read_ranked_pairs(path, *, topk):
    """Read at most topk directed matches per query and unique undirected pairs.

    Preserve source ranking order. A missing file is an error; callers distinguish
    unavailable results from a present, legitimately empty result explicitly.
    """
    if isinstance(topk, bool) or not isinstance(topk, Integral) or topk < 1:
        raise ValueError("topk must be a positive integer")
    ranked, counts = [], {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Invalid ranked pair at {path}:{number}")
        query, reference = fields
        if query == reference:
            continue
        if counts.get(query, 0) < topk:
            ranked.append((query, reference))
            counts[query] = counts.get(query, 0) + 1
    unique = sorted({tuple(sorted(pair)) for pair in ranked})
    return ranked, unique, counts


def validate_sift_database(connection, expected):
    """Check SQLite integrity and aligned keypoint/uint8 descriptor rows per image.

    ``expected`` is the saved extraction identity, including the sorted image-name
    digest and image count. No new interpretation of COLMAP feature values occurs.
    """
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise ValueError("COLMAP SQLite integrity check failed")
    names = []
    total = 0
    for name, rows, cols, size, points, dimensions, point_size in connection.execute(
        "SELECT i.name,d.rows,d.cols,length(d.data),k.rows,k.cols,length(k.data) "
        "FROM images i JOIN descriptors d ON i.image_id=d.image_id "
        "JOIN keypoints k ON i.image_id=k.image_id ORDER BY i.name"
    ):
        if (
            cols != 128
            or rows != points
            or size != rows * cols
            or dimensions not in (2, 4, 6)
            or point_size != points * dimensions * 4
        ):
            raise ValueError(f"Invalid COLMAP SIFT arrays: {name}")
        names.append(name)
        total += rows
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    if (
        not names
        or len(names) != expected["images"]
        or digest != expected["names_sha256"]
        or connection.execute("SELECT count(*) FROM images").fetchone()[0] != len(names)
    ):
        raise ValueError("COLMAP image rows differ from the extraction record")
    return {"images": len(names), "descriptors": total, "names_sha256": digest}


def sift_identity(saved, *, data_name, method_name="sift"):
    """Describe fixed SIFT inputs/settings without paths or execution resources."""
    fields = (
        "names_sha256",
        "images",
        "max_image_size",
        "max_num_features",
        "resolution",
        "first_octave",
        "descriptor_normalization",
    )
    settings = {name: saved[name] for name in fields}
    if saved["descriptor_normalization"] != "L1_ROOT":
        raise ValueError("Only documented COLMAP L1_ROOT SIFT is supported")
    return {
        "schema_version": 4,
        "kind": "local_features",
        "source": data_name,
        "selection": method_name,
        "calculation": {"algorithm": "colmap_sift_v1", **settings},
    }


def prepare_colmap_sift(settings, *, data_name, store, producer):
    """Run producer(work_database) only on a miss, under the shared result lock.

    The producer writes a closed COLMAP database and adjacent sift.json. The
    workspace is private; publication checkpoints WAL and validates row layout.
    """
    identity = sift_identity(settings, data_name=data_name)
    with store.write(identity, resume=False) as tx:
        hit = tx["hit"]
        if not hit:
            work = tx["path"] / "work"
            work.mkdir()
            source_path = work / "database.db"
            producer(source_path)
            record = json.loads((work / "sift.json").read_text())
            if sift_identity(record["identity"], data_name=data_name) != identity:
                raise ValueError("Actual SIFT settings differ from requested settings")
            with closing(sqlite3.connect(source_path)) as source:
                with closing(
                    sqlite3.connect(tx["path"] / "database.db")
                ) as destination:
                    source.backup(destination, pages=8192)
                    destination.execute("PRAGMA journal_mode=DELETE")
                    validation = validate_sift_database(destination, settings)
            atomic_json(tx["path"] / "sift.json", record)
            shutil.rmtree(work)
            path = store.commit(
                tx,
                [
                    {"path": "database.db", "format": "colmap_sift_sqlite"},
                    {"path": "sift.json", "format": "json"},
                ],
                metadata={"producer": "prepare_colmap_sift", "validation": validation},
            )
        else:
            path = tx["path"]
    manifest = store.verify(path)
    return {
        "database": str(path / "database.db"),
        "manifest": str(path / "manifest.json"),
        "artifact_key": manifest["artifact_key"],
        "cache_hit": hit,
        "reference": store.reference(path),
    }


def publish_colmap_sift(database, *, data_name, store, method_name="sift"):
    """Snapshot a documented database, including WAL, and return its actual path.

    The adjacent sift.json must establish extraction settings and fixed dataset
    version membership. Historical records establish provenance; this function
    does not invent unavailable image-content hashes or re-extract descriptors.
    SQLite backup produces a consistent standalone database without moving or
    modifying the source. Match/reconstruction work must use a separate copy.
    """
    database = Path(database).resolve()
    record = json.loads((database.parent / "sift.json").read_text())
    saved = record["identity"]
    identity = sift_identity(saved, data_name=data_name, method_name=method_name)
    with store.write(identity, resume=False) as tx:
        hit = tx["hit"]
        if not hit:
            target = tx["path"] / "database.db"
            with closing(
                sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            ) as source:
                with closing(sqlite3.connect(target)) as destination:
                    source.backup(destination, pages=8192)
                    destination.execute("PRAGMA journal_mode=DELETE")
                    validation = validate_sift_database(destination, saved)
            atomic_json(tx["path"] / "sift.json", record)
            path = store.commit(
                tx,
                [
                    {"path": "database.db", "format": "colmap_sift_sqlite"},
                    {"path": "sift.json", "format": "json"},
                ],
                metadata={
                    "producer": "gmlab.datasets.utils.sift_io.publish_colmap_sift",
                    "origin_database": str(database),
                    "input_evidence": "fixed_dataset_version_and_saved_extraction_record",
                    "validation": validation,
                },
            )
        else:
            path = tx["path"]
    manifest = store.verify(path)
    return {
        "database": str(path / "database.db"),
        "manifest": str(path / "manifest.json"),
        "artifact_key": manifest["artifact_key"],
        "cache_hit": hit,
        "reference": store.reference(path),
    }


def bow_artifact_identity(database, vocabulary, store):
    parent = store.verify(database.parent)
    return {
        "schema_version": 4,
        "kind": "indexes",
        "source": parent["identity"]["source"],
        "selection": "sift_bow_colmap",
        "calculation": {
            "algorithm": "prepared_bow_v1",
            "local_features": parent["artifact_key"],
            "vocabulary_sha256": file_digest(vocabulary),
            "database_neighbors": 1,
            "query_neighbors": 5,
            "num_checks": 64,
        },
    }


def publish_bow_cache(directory, *, database, vocabulary, store):
    """Validate a completed native index/encoding bundle and publish unchanged bytes."""
    import struct

    identity = bow_artifact_identity(database, vocabulary, store)
    source = json.loads((database.parent / "sift.json").read_text())["identity"]
    names = [
        "index.bin",
        "prepared_v1/database.bin",
        "prepared_v1/queries.bin",
        "prepared_v1/encode.json",
        "prepared_v1/build.json",
    ]
    for stage in ("encode", "build"):
        record = json.loads((directory / "prepared_v1" / (stage + ".json")).read_text())
        saved = record["identity"]
        if (
            record["images"] != source["images"]
            or saved["sift_identity"] != source
            or any(
                saved.get(k) != identity["calculation"][k]
                for k in (
                    "vocabulary_sha256",
                    "database_neighbors",
                    "query_neighbors",
                    "num_checks",
                )
            )
            or saved.get("encoding_protocol") != "prepared_bow_v1"
        ):
            raise ValueError(
                "Prepared BoW provenance differs from the requested inputs"
            )
    for filename in ("database.bin", "queries.bin"):
        with (directory / "prepared_v1" / filename).open("rb") as stream:
            if struct.unpack("<QQ", stream.read(16)) != (
                0x474D4C424F573031,
                source["images"],
            ):
                raise ValueError("Prepared BoW header differs")
    with store.write(identity, resume=False) as tx:
        if tx["hit"]:
            return tx["path"]
        for name in names:
            target = tx["path"] / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / name, target)
            if file_digest(directory / name) != file_digest(target):
                raise ValueError("BoW source changed during publication")
        return store.commit(
            tx,
            [{"path": name} for name in names],
            metadata={
                "producer": "gmlab.datasets.utils.sift_io.publish_bow_cache",
                "source_directory": str(directory),
            },
        )
