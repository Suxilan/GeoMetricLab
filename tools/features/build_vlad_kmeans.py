#!/usr/bin/env python3
"""Explicitly fit a VLAD vocabulary from one verified spatial feature stream."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gmlab.artifacts import ArtifactStore
from gmlab.datasets.utils.feature_io import open_feature_stream, iter_feature_stream
from gmlab.retrieval import spatial_descriptors, sample_descriptor_chunks, fit_kmeans
from gmlab.utils.io import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--extraction-record", type=Path)
    source.add_argument("--features", type=Path, help="Manifest-backed spatial H5")
    parser.add_argument("--stream", help="Select the named spatial output")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--num-clusters", type=int, required=True)
    parser.add_argument(
        "--max-descriptors",
        type=int,
        required=True,
        help="Explicit global sampling budget",
    )
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--max-points-per-centroid", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Explicit execution; default only previews inputs",
    )
    parser.add_argument("--out", type=Path, help="Optional explicit centers export")
    parser.add_argument("--record-output", type=Path)
    args = parser.parse_args(argv)
    for name in (
        "num_clusters",
        "max_descriptors",
        "iterations",
        "max_points_per_centroid",
        "chunk_rows",
        "threads",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    with open_feature_stream(
        args.features,
        record=args.extraction_record,
        stream=args.stream,
        cache_root=args.cache_root,
    ) as (data, _):
        source = dict(data)
    if len(source["shape"]) != 4:
        raise ValueError("Spatial KMeans requires verified [N,C,H,W] feature maps")
    images, channels, height, width = source["shape"]
    sample_rows = min(args.max_descriptors, images * height * width)
    summary = {
        "artifact_key": source["manifest"]["artifact_key"],
        "shape": source["shape"],
        "sample_rows": sample_rows,
        "clusters": args.num_clusters,
        "normalize": args.normalize,
        "can_fit": sample_rows >= args.num_clusters,
    }
    if not args.fit:
        print(json.dumps(summary, indent=2))
        return summary
    if not summary["can_fit"]:
        raise ValueError("Not enough sampled patches for the requested clusters")
    input_directory = Path(source["path"]).parent
    if any(
        path and path.resolve().is_relative_to(input_directory)
        for path in (args.out, args.record_output)
    ):
        raise ValueError(
            "Vocabulary outputs must be outside the source feature artifact"
        )
    if (
        args.out
        and args.record_output
        and args.out.resolve() == args.record_output.resolve()
    ):
        raise ValueError("Vocabulary and run record need distinct paths")
    if (
        args.extraction_record
        and args.record_output
        and args.extraction_record.resolve() == args.record_output.resolve()
    ):
        raise ValueError("Run record must not overwrite the source extraction record")
    parent = source["manifest"]
    reference = {"artifact_key": parent["artifact_key"], "payloads": parent["payloads"]}
    identity = {
        "schema_version": 4,
        "kind": "vocabularies",
        "source": parent["identity"]["source"],
        "selection": "spatial_vlad_kmeans",
        "calculation": {
            "algorithm": "faiss_kmeans_spatial_v1",
            "parent": reference,
            "sampling": "uniform_random_priority_rows_v1",
            "max_descriptors": args.max_descriptors,
            "clusters": args.num_clusters,
            "iterations": args.iterations,
            "max_points_per_centroid": args.max_points_per_centroid,
            "seed": args.seed,
            "normalize": args.normalize,
        },
    }
    store = ArtifactStore(args.cache_root)
    transaction = (
        store.export(identity, args.out)
        if args.out
        else store.write(identity, resume=False)
    )
    with transaction as tx:
        hit = tx["hit"]
        filename = tx.get("filename", "centers.npy")
        if not hit:
            chunks = (
                spatial_descriptors(values)
                for _, values in iter_feature_stream(
                    source["record"],
                    chunk_rows=args.chunk_rows,
                    cache_root=args.cache_root,
                )
            )
            sampled = sample_descriptor_chunks(
                chunks,
                max_descriptors=args.max_descriptors,
                seed=args.seed,
                normalize=args.normalize,
            )
            import faiss

            faiss.omp_set_num_threads(args.threads)
            centers = fit_kmeans(
                sampled,
                num_clusters=args.num_clusters,
                iterations=args.iterations,
                seed=args.seed,
                min_points_per_centroid=1,
                max_points_per_centroid=args.max_points_per_centroid,
            )
            if (
                centers.shape != (args.num_clusters, channels)
                or not np.isfinite(centers).all()
            ):
                raise ValueError("Invalid fitted vocabulary centers")
            with (tx["path"] / filename).open("xb") as file:
                np.save(file, centers)
            path = store.commit(
                tx,
                [
                    {
                        "path": filename,
                        "format": "numpy",
                        "shape": list(centers.shape),
                        "dtype": str(centers.dtype),
                    }
                ],
                parents=[reference],
                metadata={
                    "sampled_rows": len(sampled),
                    "threads": args.threads,
                    "backend": "faiss_cpu",
                },
            )
        else:
            path = tx["path"]
    manifest = store.verify(path)
    result = {
        "vocabulary": str(path / filename),
        "manifest": str(path / "manifest.json"),
        "artifact_key": manifest["artifact_key"],
        "cache_hit": hit,
    }
    if path.is_relative_to(store.root):
        result["reference"] = store.reference(path)
    if args.record_output:
        if args.record_output.resolve() in {
            (path / filename).resolve(),
            (path / "manifest.json").resolve(),
        }:
            raise ValueError("Run record must not overwrite the vocabulary artifact")
        atomic_json(args.record_output, result)
    print(json.dumps({**summary, "result": result}, indent=2))
    return result


if __name__ == "__main__":
    main()
