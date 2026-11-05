#!/usr/bin/env python3
"""Retrieve Top-K from existing global H5 descriptors and export unique COLMAP pairs.

Extraction belongs to tools/features/extract_image_features.py.
Optional descriptor reranking runs before Top-K pair construction.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gmlab.datasets.utils.feature_io import read_global_features

from gmlab.retrieval import topk_neighbors, build_colmap_pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--features",
        "--cache",
        type=Path,
        dest="features",
        help="Explicit legacy input (unverified)",
    )
    source.add_argument(
        "--extraction-record",
        type=Path,
        help="Verified feature location returned by extraction",
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Image root used to verify cache names"
    )
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument(
        "--reranking-config", type=Path, help="YAML {method: name, ...parameters}"
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--stream", help="Select a global stream from a multi-stream record"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefer-gpu",
        "--prefer_gpu_faiss",
        dest="prefer_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer GPU FAISS when auto search selects FAISS (default: enabled)",
    )
    args = parser.parse_args(argv)
    if args.topk <= 0:
        parser.error("--topk must be positive")
    root = args.root.resolve()
    data = read_global_features(
        args.features,
        record=args.extraction_record,
        cache_root=args.cache_root,
        stream=args.stream,
        allow_unverified=args.features is not None,
    )
    names, values = data["image_names"], np.asarray(data["features"], dtype=np.float32)
    if values.ndim != 2 or len(values) != len(names) or not np.isfinite(values).all():
        raise ValueError(
            "Expected finite global descriptors [N,D] matching image_names"
        )
    if len(set(names)) != len(names):
        raise ValueError("SfM feature cache contains repeated images")
    if np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Feature cache contains zero descriptors")
    for name in names:
        path = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file():
            raise ValueError(f"Cache name is not an image relative to --root: {name}")
    neighbors = topk_neighbors(
        torch.from_numpy(values),
        args.topk,
        args.prefer_gpu,
        reranking=(
            yaml.safe_load(args.reranking_config.read_text())
            if args.reranking_config
            else None
        ),
    )
    pairs = build_colmap_pairs(names, neighbors, args.topk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for a, b in pairs:
            stream.write(f"{a} {b}\n")
    print(
        f"{len(names):,} images; {len(pairs):,} unique undirected pairs: {args.output}"
    )
    return pairs


if __name__ == "__main__":
    main()
