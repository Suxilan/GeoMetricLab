#!/usr/bin/env python3
"""Rank verified query/gallery global feature records without requiring GT."""

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gmlab.datasets.utils.feature_io import read_global_features
from gmlab.retrieval import retrieve_chunks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-record", type=Path, required=True)
    parser.add_argument("--gallery-record", type=Path, required=True)
    parser.add_argument("--query-stream")
    parser.add_argument("--gallery-stream")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--config", type=Path, help="YAML with search and reranking sections"
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, required=True, help="Ranked JSONL results"
    )
    args = parser.parse_args(argv)
    config = {} if args.config is None else yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict) or config.keys() - {"search", "reranking"}:
        raise ValueError("Retrieval config accepts only search and reranking")
    query = read_global_features(
        record=args.query_record, stream=args.query_stream, cache_root=args.cache_root
    )
    gallery = read_global_features(
        record=args.gallery_record,
        stream=args.gallery_stream,
        cache_root=args.cache_root,
    )
    rankings = retrieve_chunks(
        query["features"],
        gallery["features"],
        k=args.topk,
        reranking=config.get("reranking"),
        **config.get("search", {}),
    )
    # Validate and execute the first chunk before creating the output file.
    first = next(rankings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from itertools import chain

    with args.output.open("w") as stream:
        stream.write(
            json.dumps(
                {
                    "type": "metadata",
                    "schema_version": 1,
                    "query": query["record"],
                    "gallery": gallery["record"],
                    "topk": args.topk,
                    "config": config,
                }
            )
            + "\n"
        )
        for start, rows in chain([first], rankings):
            for offset, positions in enumerate(rows.tolist()):
                index = start + offset
                stream.write(
                    json.dumps(
                        {
                            "type": "ranking",
                            "query_index": index,
                            "query_name": query["image_names"][index],
                            "gallery_indices": positions,
                            "gallery_names": [
                                gallery["image_names"][i] for i in positions
                            ],
                        }
                    )
                    + "\n"
                )
    return args.output


if __name__ == "__main__":
    main()
