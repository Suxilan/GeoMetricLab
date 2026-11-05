#!/usr/bin/env python3
"""Display actual cached preprocessing; read-only, no payload hash verification."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gmlab.artifacts import ArtifactStore
from gmlab.datasets.utils.feature_description import describe_feature


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--data-name")
    parser.add_argument("--method")
    parser.add_argument(
        "--json", action="store_true", help="Include the full ordered preprocessing"
    )
    args = parser.parse_args(argv)
    rows = []
    for manifest in ArtifactStore(args.cache_root).inventory():
        identity = manifest["identity"]
        if identity["kind"] not in (
            "global_features",
            "spatial_features",
            "local_features",
        ):
            continue
        if args.data_name and identity["source"] != args.data_name:
            continue
        if args.method and identity["selection"] != args.method:
            continue
        rows.append({"path": manifest["path"], **describe_feature(manifest)})
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print("DATA\tMETHOD\tKEY\tRESIZE\tOUTPUT\tPATH")
        for row in rows:
            print(
                "\t".join(
                    str(v)
                    for v in (
                        row["data_name"],
                        row["method"],
                        row["artifact_key"][:16],
                        row["resize"],
                        row["output"],
                        row["path"],
                    )
                )
            )
    return rows


if __name__ == "__main__":
    main()
