#!/usr/bin/env python3
"""COLMAP SIFT/BoW and SIFT-VLAD64 stages; shell composes independent commands.

SIFT can validate an existing SfM cache or extract at original resolution.
Encoding caches and timings have separate identities; training uses a separate database.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gmlab.datasets.utils.image_list import find_images
from gmlab.backends import colmap as colmap_backend
from gmlab.backends.colmap import colmap_path
from gmlab.datasets.utils.sift_io import (
    bow_artifact_identity,
    descriptor_rows,
    local_descriptors,
    iter_sift_descriptors,
    sift_database_summary,
    sift_image_names,
    read_ranked_pairs,
    publish_bow_cache,
    read_database,
    read_sift_record,
)
from gmlab.retrieval import aggregate_vlad, fit_kmeans, sample_local_descriptors
from gmlab.artifacts import file_digest as digest
from gmlab.utils.io import atomic_json as save_json
from gmlab.utils.resources import monitor_process

TIMING_PROTOCOL = "retrieval_stages_v3"


def timing_profile(threads):
    return dict(
        timing_protocol=TIMING_PROTOCOL,
        threads=threads,
        cpu_affinity=sorted(os.sched_getaffinity(0)),
    )


def reusable_timing(record, profile):
    """Allow exact original CPU budgets for recovered classical stage records."""
    saved = record.get("resource_profile", {})
    if saved == profile:
        return True
    return bool(
        record.get("recovered_from_codex")
        and saved.get("timing_protocol") == "retrieval_stages_v2"
        and all(
            saved.get(key) == profile.get(key) for key in ("threads", "cpu_affinity")
        )
    )


def build_prepared_colmap(colmap):
    return colmap_backend.build_prepared_colmap(
        colmap,
        adapter_sources=Path(__file__).parent / "native",
        source_root=os.environ.get("COLMAP_SOURCE"),
    )


def run_logged(
    command, log, extra_environment=None, resource_output=None, append=False
):
    return colmap_backend.run_colmap(
        command,
        log,
        environment=extra_environment,
        monitor=monitor_process(os.getpid(), resource_output),
        append=append,
    )


def _sift_workspace(args):
    from PIL import Image

    paths = find_images([args.root])
    if args.limit:
        indices = sorted(
            np.random.default_rng(args.seed).choice(
                len(paths), min(args.limit, len(paths)), replace=False
            )
        )
        paths = [paths[i] for i in indices]
    names = [p.relative_to(args.root).as_posix() for p in paths]
    if not names:
        raise ValueError("No images found")
    if any("\n" in name or "\r" in name for name in names):
        raise ValueError("Image list cannot contain newline characters")
    relocated_from = getattr(args, "relocated_from", None)
    if relocated_from is not None and not args.reuse_only:
        raise ValueError("--relocated-from requires --reuse-only")
    if args.reuse_only:
        manifest = args.database.parent / "sift.json"
        if not manifest.is_file() or not args.database.is_file():
            raise FileNotFoundError(f"Prepared SIFT cache required: {args.database}")
        saved = json.loads(manifest.read_text())["identity"]
        names_sha = hashlib.sha256("\n".join(names).encode()).hexdigest()
        source_root = relocated_from if relocated_from is not None else args.root
        if (
            saved["image_root"] != str(source_root.resolve())
            or saved["names_sha256"] != names_sha
            or saved["images"] != len(names)
        ):
            raise ValueError(
                "Prepared SIFT cache does not match the requested image set"
            )
        with closing(read_database(args.database)) as connection:
            found = sift_image_names(connection, require_keypoints=True)
        if found != names:
            raise ValueError(
                "Prepared SIFT database has missing or extra images/features"
            )
        print(
            f"Reuse SIFT (max side {saved['max_image_size']}): {args.database}",
            flush=True,
        )
        return
    max_dimension = 0
    for path in paths:
        with Image.open(path) as image:
            max_dimension = max(max_dimension, *image.size)
    identity = dict(
        image_root=str(args.root.resolve()),
        names_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest(),
        images=len(names),
        max_image_size=max_dimension,
        resolution="original",
        max_num_features=args.max_features,
        first_octave=0,
        descriptor_normalization="L1_ROOT",
        seed=args.seed,
        colmap=colmap_path(args.colmap),
        gpu=args.gpu,
        threads=args.threads,
    )
    if getattr(args, "plan_only", False):
        return identity
    output = args.database.parent
    manifest = output / "sift.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if (
            any(
                saved["identity"].get(k) != identity.get(k)
                for k in (
                    "names_sha256",
                    "images",
                    "max_image_size",
                    "max_num_features",
                    "resolution",
                    "first_octave",
                    "descriptor_normalization",
                    "seed",
                )
            )
            or not args.database.is_file()
        ):
            raise ValueError("SIFT settings changed; choose a new cache directory")
        print(f"Reuse {args.database}", flush=True)
        return
    if args.database.exists():
        raise ValueError(
            "Incomplete/untracked database: use a new output to avoid timing resumed work"
        )
    output.mkdir(parents=True, exist_ok=True)
    image_list = output / "images.txt"
    image_list.write_text("\n".join(names) + "\n")
    command = colmap_backend.sift_command(
        identity["colmap"],
        database=args.database,
        image_root=args.root,
        image_list=image_list,
        max_image_size=max_dimension,
        max_features=args.max_features,
        gpu=args.gpu,
        threads=args.threads,
    )
    seconds = run_logged(command, output / "sift.log")
    with closing(read_database(args.database)) as connection:
        summary = sift_database_summary(connection, names)
    save_json(
        manifest,
        dict(
            stage="sift_extraction",
            identity=identity,
            seconds=seconds,
            images_per_second=len(names) / seconds,
            **summary,
            command=command,
            scope="process wall time including CUDA initialization, image decoding and database writes; no resize",
        ),
    )
    print(f"SIFT: {len(names)} images in {seconds:.2f}s; {args.database}", flush=True)


def sift(args):
    """Validate/prepare with the existing algorithm, then publish via gmlab."""
    from gmlab.artifacts import ArtifactStore
    from gmlab.utils.io import atomic_json
    from gmlab.datasets.utils.sift_io import publish_colmap_sift

    from copy import copy
    from gmlab.datasets.utils.sift_io import prepare_colmap_sift

    store = ArtifactStore(args.cache_root)
    if args.database is None and args.reuse_only:
        path = store.resolve(
            "local_features", args.data_name, "sift", artifact_key=args.artifact_key
        )
        args.database = path / "database.db"
    if (args.database is not None and args.database.is_file()) or args.reuse_only:
        _sift_workspace(args)
        result = publish_colmap_sift(
            args.database, data_name=args.data_name, store=store
        )
    else:
        planned = copy(args)
        planned.plan_only = True
        settings = _sift_workspace(planned)

        def produce(database):
            workspace = copy(args)
            workspace.database = database
            _sift_workspace(workspace)

        result = prepare_colmap_sift(
            settings, data_name=args.data_name, store=store, producer=produce
        )
    if args.record_output:
        atomic_json(args.record_output, result)
    print(json.dumps(result, indent=2))
    return result


def _train_vlad_workspace(args):
    import faiss

    if args.output.exists():
        raise FileExistsError(args.output)
    faiss.omp_set_num_threads(args.threads)
    started = time.perf_counter()
    with closing(read_database(args.database)) as connection:
        names = sift_image_names(connection)
        values = sample_local_descriptors(
            (values for _, values in iter_sift_descriptors(connection)),
            per_image=args.per_image,
            seed=args.seed,
        )
    centers = fit_kmeans(values, num_clusters=64, iterations=50, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, centers)
    save_json(
        args.output.with_suffix(".json"),
        dict(
            database=str(args.database),
            database_manifest=json.loads(
                (args.database.parent / "sift.json").read_text()
            ),
            image_count=len(names),
            descriptor_count=len(values),
            seed=args.seed,
            centers=64,
            dimension=128,
            seconds=time.perf_counter() - started,
            sha256=digest(args.output),
            normalization="quantized COLMAP L1_ROOT then L2",
        ),
    )
    print(f"Trained SIFT VLAD64 vocabulary: {args.output}", flush=True)


def train_vlad(args):
    from copy import copy
    from gmlab.artifacts import ArtifactStore
    from gmlab.utils.io import atomic_json

    store = ArtifactStore(args.cache_root)
    parent = store.verify(args.database.parent)
    identity = {
        "schema_version": 4,
        "kind": "vocabularies",
        "source": parent["identity"]["source"],
        "selection": "sift_vlad64",
        "calculation": {
            "algorithm": "sift_vlad_kmeans_v1",
            "parent": parent["artifact_key"],
            "seed": args.seed,
            "per_image": args.per_image,
            "centers": 64,
            "dimension": 128,
            "niter": 50,
            "nredo": 1,
            "normalization": "quantized COLMAP L1_ROOT then L2",
        },
    }
    transaction = (
        store.export(identity, args.output) if args.output else store.write(identity)
    )
    with transaction as tx:
        hit = tx["hit"]
        filename = tx.get("filename", "centers.npy")
        if not hit:
            workspace = copy(args)
            workspace.output = tx["path"] / filename
            _train_vlad_workspace(workspace)
            centers = np.load(workspace.output)
            if centers.shape != (64, 128) or not np.isfinite(centers).all():
                raise ValueError("Invalid SIFT vocabulary")
            path = store.commit(
                tx,
                [
                    {"path": filename, "format": "numpy", "shape": [64, 128]},
                    {
                        "path": Path(filename).with_suffix(".json").name,
                        "format": "json",
                    },
                ],
                metadata={"producer": "classical_retrieval.train_vlad"},
                parents=[
                    {
                        "artifact_key": parent["artifact_key"],
                        "payloads": parent["payloads"],
                    }
                ],
            )
        else:
            path = tx["path"]
    result = {
        "vocabulary": str(path / filename),
        "manifest": str(path / "manifest.json"),
        "artifact_key": store.key(identity),
        "cache_hit": hit,
    }
    if path.is_relative_to(store.root):
        result["reference"] = store.reference(path)
    if args.record_output:
        atomic_json(args.record_output, result)
    print(json.dumps(result, indent=2))
    return result


def encode_vlad(args):
    import faiss
    from contextlib import closing
    from gmlab.artifacts import ArtifactStore
    from gmlab.utils.io import atomic_json
    from gmlab.datasets.utils.feature_io import (
        GlobalFeatureWriter,
        validate_global_features,
        write_samples,
    )

    store = ArtifactStore(args.cache_root)
    parent = store.verify(args.database.parent)
    centers = np.load(args.vocabulary)
    if centers.shape != (64, 128) or not np.isfinite(centers).all():
        raise ValueError("Expected finite 64x128 SIFT vocabulary")
    source = json.loads((args.database.parent / "sift.json").read_text())["identity"]
    training = json.loads(args.vocabulary.with_suffix(".json").read_text())
    if (
        source["descriptor_normalization"]
        != training["database_manifest"]["identity"]["descriptor_normalization"]
    ):
        raise ValueError("SIFT vocabulary/query normalization differs")
    identity = {
        "schema_version": 4,
        "kind": "global_features",
        "source": parent["identity"]["source"],
        "selection": "sift_vlad64",
        "calculation": {
            "algorithm": "sift_vlad64_intranorm_l2_v1",
            "local_features": parent["artifact_key"],
            "vocabulary_sha256": digest(args.vocabulary),
            "dtype": "float32",
            "dimension": 8192,
        },
    }
    transaction = (
        store.export(identity, args.output) if args.output else store.write(identity)
    )
    started = time.perf_counter()
    with transaction as tx:
        hit = tx["hit"]
        filename = tx.get("filename", "features.h5")
        if not hit:
            faiss.omp_set_num_threads(args.threads)
            index = faiss.IndexFlatL2(128)
            index.add(centers)
            with closing(read_database(args.database)) as connection:
                names = sift_image_names(connection, require_descriptors=False)
                samples = [
                    {"row_id": i, "sample_id": name} for i, name in enumerate(names)
                ]
                write_samples(tx["path"] / "samples.jsonl", samples)
                writer = GlobalFeatureWriter(tx["path"] / filename, samples)
                try:
                    for i, (name, rows, cols, blob) in enumerate(
                        descriptor_rows(connection)
                    ):
                        if name != names[i]:
                            raise ValueError("SIFT descriptor ordering changed")
                        vector = aggregate_vlad(
                            local_descriptors(rows, cols, blob), centers, index
                        )
                        writer.write(np.asarray([i]), vector[None])
                        if (i + 1) % 256 == 0:
                            print(f"Encoding VLAD: {i + 1}/{len(names)}", flush=True)
                finally:
                    writer.close()
            schema = validate_global_features(tx["path"] / filename, len(names))
            path = store.commit(
                tx,
                [schema, {"path": "samples.jsonl", "format": "jsonl"}],
                metadata={"producer": "classical_retrieval.encode_vlad"},
                parents=[
                    {
                        "artifact_key": parent["artifact_key"],
                        "payloads": parent["payloads"],
                    }
                ],
            )
        else:
            path = tx["path"]
    manifest = store.verify(path)
    payload = next(p for p in manifest["payloads"] if p["path"] == filename)
    result = {
        "features": str(path / filename),
        "manifest": str(path / "manifest.json"),
        "artifact_key": manifest["artifact_key"],
        "features_sha256": payload["sha256"],
        "shape": payload["shape"],
        "dtype": payload["dtype"],
        "samples": str(path / "samples.jsonl"),
        "reference": {
            **(
                store.reference(path)
                if path.is_relative_to(store.root)
                else {
                    "artifact_key": manifest["artifact_key"],
                    "payloads": manifest["payloads"],
                }
            ),
            "feature_payload": filename,
        },
        "cache_hit": hit,
        "seconds": time.perf_counter() - started,
    }
    if args.record_output:
        atomic_json(args.record_output, result)
    print(json.dumps(result, indent=2))
    return result


def save_bow_summary(
    args, identity, profile, indexed, prepared, names, encoding, building, query, native
):
    """Save completed timing rounds; missing pairs or finalization stay explicit."""
    pairs_path = prepared / "ranked_pairs.txt"
    pairs_available = pairs_path.is_file()
    ranked, unique, counts = (
        read_ranked_pairs(pairs_path, topk=args.topk)
        if pairs_available
        else ([], [], {})
    )
    if pairs_available:
        (args.output / "ranked_pairs.txt").write_text(
            "".join(f"{a} {b}\n" for a, b in ranked)
        )
    if pairs_available:
        (args.output / "colmap_pairs.txt").write_text(
            "".join(f"{a} {b}\n" for a, b in unique)
        )
    trials = query["query_trials_seconds"]
    seconds = float(np.median(trials))
    save_json(
        args.output / "bow.json",
        dict(
            **profile,
            method="sift_bow_colmap",
            identity=identity,
            blas_threads=1,
            query_inner_threads=1,
            image_names_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest(),
            images=len(names),
            topk=min(args.topk, len(names) - 1),
            repeats=len(trials),
            requested_repeats=args.repeats,
            pairs_available=pairs_available,
            query_complete=query.get("complete", True),
            warmup_queries=query.get("warmup_queries"),
            representation_seconds=encoding["compute_seconds"],
            representation_pipeline_seconds=encoding["stage_wall_seconds"],
            representation_finalize_seconds=query.get("query_finalize_seconds"),
            encoding_compute_seconds=encoding["compute_seconds"],
            index_seconds=building["compute_seconds"],
            index_path=str(indexed),
            index_bytes=indexed.stat().st_size,
            query_trials_seconds=trials,
            query_seconds=seconds,
            recovered_from_log=query.get("recovered_from_log", False),
            queries_per_second=len(names) / seconds,
            unique_pairs=len(unique) if pairs_available else None,
            queries_fewer_than_topk=(
                len(names)
                - sum(v == min(args.topk, len(names) - 1) for v in counts.values())
            )
            if pairs_available
            else None,
            encoding=encoding,
            building=building,
            query=query,
            native_binary=str(native),
            scope=(
                "COLMAP BoW+HE; cached word assignments and binary signatures; "
                "pure batch-wall retrieval includes inverted voting, normalization and ranking; "
                "no SIFT reads, quantization or spatial verification inside query timer"
            ),
        ),
    )
    print(
        f"BoW pure retrieval: {seconds:.3f}s, {len(names) / seconds:.1f} queries/s",
        flush=True,
    )


def _save_index_record(store, published, identity, output, record_output=None):
    result = {
        "index": str(published / "index.bin"),
        "artifact_key": store.key(identity),
        "manifest": str(published / "manifest.json"),
        "reference": store.reference(published),
    }
    save_json(output / "index.json", result)
    if record_output:
        save_json(record_output, result)
    return result


def bow(args):
    """Cache native word/HE encoding, then time index and prepared queries."""
    from gmlab.artifacts import ArtifactStore

    args.output.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(args.cache_root)
    artifact_identity = bow_artifact_identity(args.database, args.vocabulary, store)
    published = store.lookup(artifact_identity)
    vocabulary_sha = digest(args.vocabulary)
    cache_dir = published or args.index_dir or (args.output / "build")
    prepared = cache_dir / "prepared_v1"
    if published is None:
        prepared.mkdir(parents=True, exist_ok=True)
    indexed = cache_dir / "index.bin"
    source = json.loads((args.database.parent / "sift.json").read_text())["identity"]
    identity = dict(
        database=str(args.database.resolve()),
        sift_identity=source,
        vocabulary_sha256=vocabulary_sha,
        encoding_protocol="prepared_bow_v1",
        database_neighbors=1,
        query_neighbors=5,
        num_checks=64,
    )
    profile = timing_profile(args.threads)
    manifest = args.output / "bow.json"
    force_query = getattr(args, "force_query", False)
    if published is not None and manifest.is_file() and not force_query:
        saved = json.loads(manifest.read_text())
        if (
            saved.get("identity") == identity
            and all(saved.get(k) == v for k, v in profile.items())
            and saved.get("topk") == min(args.topk, source["images"] - 1)
            and len(saved.get("query_trials_seconds", [])) >= args.repeats
        ):
            print(
                f"Reuse completed BoW timing: {manifest}; "
                f"pairs available: {saved.get('pairs_available', True)}",
                flush=True,
            )
            return _save_index_record(
                store,
                published,
                artifact_identity,
                args.output,
                getattr(args, "record_output", None),
            )
    native = build_prepared_colmap(colmap_path(args.colmap))
    base = colmap_backend.prepared_bow_command(
        native,
        database=args.database,
        prepared=prepared,
        threads=args.threads,
        topk=args.topk,
        repeats=args.repeats,
    )
    with closing(read_database(args.database)) as connection:
        names = sift_image_names(connection, require_descriptors=False)
    # Outer query parallelism receives the CPU budget; inner BLAS stays single-threaded.
    environment = dict(
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS=str(args.threads)
    )

    def phase(stage, vocabulary, artifacts):
        path = prepared / f"{stage}.json"
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("identity") != identity:
                raise ValueError(f"BoW cache source changed: {path}")
            if reusable_timing(saved, profile) and all(p.is_file() for p in artifacts):
                print(f"Reuse BoW {stage}: {path}", flush=True)
                return saved
        print(
            f"BoW {stage}: {len(names)} images, {args.threads} CPU threads", flush=True
        )
        temporary_metrics = prepared / f"{stage}.partial.json"
        command = base + [
            "--benchmark_stage",
            stage,
            "--vocab_tree_path",
            str(vocabulary),
            "--phase_metrics_path",
            str(temporary_metrics),
        ]
        if stage == "build":
            command += ["--output_index_path", str(indexed.with_suffix(".partial.bin"))]
        work_database = prepared / "encoding_database.db"
        if stage == "encode":
            # COLMAP opens SQLite READWRITE and changes journal mode even for
            # extraction reads. Give it a private copy of the shared snapshot.
            shutil.copyfile(args.database, work_database)
            command[command.index("--database_path") + 1] = str(work_database)
        try:
            run_logged(
                command,
                prepared / f"{stage}.log",
                environment,
                prepared / f"resources_{stage}.json",
            )
        finally:
            for suffix in ("", "-wal", "-shm"):
                Path(str(work_database) + suffix).unlink(missing_ok=True)
        value = json.loads(temporary_metrics.read_text())
        if value["images"] != len(names):
            raise ValueError(f"Incomplete BoW stage: {stage}")
        if stage == "build":
            indexed.with_suffix(".partial.bin").replace(indexed)
        value.update(identity=identity, resource_profile=profile)
        save_json(path, value)
        temporary_metrics.unlink(missing_ok=True)
        return value

    if published is None:
        encoding = phase(
            "encode",
            args.vocabulary,
            [prepared / "database.bin", prepared / "queries.bin"],
        )
        building = phase("build", args.vocabulary, [indexed])
        published = publish_bow_cache(
            cache_dir, database=args.database, vocabulary=args.vocabulary, store=store
        )
    else:
        encoding = json.loads((prepared / "encode.json").read_text())
        building = json.loads((prepared / "build.json").read_text())
    indexed = published / "index.bin"
    # Native queries write pairs next to their inputs. Only the run workspace is
    # writable; the shared encodings and index remain immutable.
    prepared = args.output / "query_inputs"
    prepared.mkdir(parents=True, exist_ok=True)
    for name in ("database.bin", "queries.bin"):
        link = prepared / name
        target = published / "prepared_v1" / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                link.unlink()
        elif link.exists():
            raise FileExistsError(link)
        if not link.exists():
            link.symlink_to(target.resolve())
    base[base.index("--prepared_path") + 1] = str(prepared)
    index_record = _save_index_record(
        store,
        published,
        artifact_identity,
        args.output,
        getattr(args, "record_output", None),
    )
    query_path = args.output / "query.json"
    context = dict(
        identity=identity,
        resource_profile=profile,
        topk=args.topk,
        index_bytes=indexed.stat().st_size,
        index_mtime_ns=indexed.stat().st_mtime_ns,
    )
    query_identity = hashlib.sha256(
        json.dumps(context, sort_keys=True).encode()
    ).hexdigest()
    previous = None
    if query_path.is_file() and not force_query:
        candidate = json.loads(query_path.read_text())
        if candidate.get("query_identity") == query_identity:
            previous = candidate
    trials = previous.get("query_trials_seconds", []) if previous else []
    if len(trials) < args.repeats:
        warmup = getattr(args, "warmup_queries", 64)
        print(
            f"BoW query: {len(trials)} measured rounds saved; "
            f"target {args.repeats}; warmup {warmup or 'all'} queries",
            flush=True,
        )
        failure = None
        try:
            run_logged(
                base
                + [
                    "--benchmark_stage",
                    "query",
                    "--vocab_tree_path",
                    str(indexed),
                    "--phase_metrics_path",
                    str(query_path),
                    "--warmup_queries",
                    str(warmup),
                    "--previous_trials",
                    ",".join(map(str, trials)),
                    "--query_identity",
                    query_identity,
                ],
                args.output / "query.log",
                environment,
                args.output / "resources_retrieval.json",
                append=bool(previous),
            )
        except RuntimeError as error:
            failure = error
        if not query_path.is_file():
            raise failure or RuntimeError("BoW did not save a query checkpoint")
        query = json.loads(query_path.read_text())
        if query.get("query_identity") != query_identity:
            raise failure or ValueError("BoW query checkpoint identity differs")
        if failure and not query.get("query_trials_seconds"):
            raise failure
    else:
        query, failure = previous, None
    if query["images"] != len(names):
        raise ValueError("Incomplete BoW query batch")
    save_bow_summary(
        args,
        identity,
        profile,
        indexed,
        prepared,
        names,
        encoding,
        building,
        query,
        native,
    )
    if failure:
        raise failure
    return index_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("sift")
    s.add_argument("--root", required=True, type=Path)
    s.add_argument("--database", type=Path, help="Explicit historical source database")
    s.add_argument(
        "--artifact-key", help="Select a full key when multiple SIFT variants exist"
    )
    s.add_argument("--data-name", required=True, help="Fixed dataset version")
    s.add_argument("--cache-root", type=Path)
    s.add_argument("--record-output", type=Path)
    s.add_argument("--max-features", type=int, default=8192)
    s.add_argument("--gpu", type=int, default=0)
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--colmap")
    s.add_argument(
        "--reuse-only",
        action="store_true",
        help="Validate a prepared cache without extracting SIFT",
    )
    s.add_argument(
        "--relocated-from",
        type=Path,
        help="Original image root recorded in a reused cache, before moving its source images",
    )
    t = sub.add_parser("train-vlad")
    t.add_argument("--database", type=Path)
    t.add_argument("--output", type=Path)
    t.add_argument("--per-image", type=int, default=256)
    t.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("encode-vlad")
    v.add_argument("--database", type=Path)
    v.add_argument("--vocabulary", required=True, type=Path)
    v.add_argument("--output", type=Path)
    b = sub.add_parser("bow")
    b.add_argument("--database", type=Path)
    b.add_argument("--vocabulary", required=True, type=Path)
    b.add_argument("--output", required=True, type=Path)
    b.add_argument(
        "--index-dir", type=Path, help="Override the task/dataset BoW index cache"
    )
    b.add_argument("--topk", type=int, default=30)
    b.add_argument("--repeats", type=int, default=1)
    b.add_argument(
        "--warmup-queries",
        type=int,
        default=64,
        help="Warmup queries; 0 means a complete warmup pass",
    )
    b.add_argument(
        "--force-query",
        action="store_true",
        help="Discard previous query timings and measure again",
    )
    b.add_argument("--colmap")
    for command in (t, v, b):
        command.add_argument("--sift-record", type=Path)
        command.add_argument("--cache-root", type=Path)
        command.add_argument("--record-output", type=Path)
    for command in (s, t, v, b):
        command.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)
    if args.stage != "sift":
        if bool(args.database) == bool(args.sift_record):
            parser.error("Choose --database or --sift-record")
        if args.sift_record:
            from gmlab.artifacts import ArtifactStore

            args.database = read_sift_record(
                args.sift_record, store=ArtifactStore(args.cache_root)
            )
    if getattr(args, "warmup_queries", 0) < 0:
        parser.error("warmup_queries must be nonnegative")
    for key in ("threads", "max_features", "limit", "per_image", "topk", "repeats"):
        if getattr(args, key, 1) is not None and getattr(args, key, 1) < 1:
            parser.error(f"{key} must be positive")
    return {
        "sift": sift,
        "train-vlad": train_vlad,
        "encode-vlad": encode_vlad,
        "bow": bow,
    }[args.stage](args)


if __name__ == "__main__":
    main()
