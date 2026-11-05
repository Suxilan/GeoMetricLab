#!/usr/bin/env python3
"""Build SUES-200 per-height and mixed-height metadata from the official split.

Protocol arrays live beside the raw data, under the dataset root.
Gallery contains all 200 scenes, queries contain only the 80 test scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SPLIT_FILE = Path(__file__).with_name("sues200_official_split.json")
HEIGHTS = (150, 200, 250, 300)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_metadata(
    dataset_root: str | Path = "data/sues200", metadata_root: str | Path | None = None
) -> list[dict]:
    root = Path(dataset_root)
    output = Path(metadata_root) if metadata_root is not None else root
    split = json.loads(SPLIT_FILE.read_text())
    train, test = split["train_ids"], split["test_ids"]
    all_ids = sorted(train + test)
    if (len(train), len(test), len(set(all_ids))) != (120, 80, 200) or set(all_ids) != {
        f"{i:04d}" for i in range(1, 201)
    }:
        raise ValueError("Invalid pinned official SUES-200 split")

    def images(directory: Path, expected: int) -> list[str]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing SUES-200 scene/height: {directory}")
        paths = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(paths) != expected:
            raise ValueError(
                f"Expected {expected} images in {directory}, found {len(paths)}"
            )
        return [p.relative_to(root).as_posix() for p in paths]

    satellite = {scene: images(root / "satellite-view" / scene, 1) for scene in all_ids}
    # Validate the entire raw release before writing any protocol files.
    drones = {
        height: {
            scene: images(root / "drone_view_512" / scene / str(height), 50)
            for scene in all_ids
        }
        for height in HEIGHTS
    }
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for height in (*HEIGHTS, None):
        drone_view = (
            drones[height]
            if height is not None
            else {
                scene: [p for h in HEIGHTS for p in drones[h][scene]]
                for scene in all_ids
            }
        )
        for direction in ("drone2satellite", "satellite2drone"):
            task = f"sues200_{direction}" + (f"_{height}" if height is not None else "")
            db_view, q_view = (
                (satellite, drone_view)
                if direction == "drone2satellite"
                else (drone_view, satellite)
            )
            db = [p for scene in all_ids for p in db_view[scene]]
            queries = [p for scene in test for p in q_view[scene]]
            db_by_id, offset = {}, 0
            for scene in all_ids:
                db_by_id[scene] = np.arange(
                    offset, offset + len(db_view[scene]), dtype=np.int64
                )
                offset += len(db_view[scene])
            gt = np.stack([db_by_id[scene] for scene in test for _ in q_view[scene]])
            arrays = {
                f"{task}_dbImages.npy": np.asarray(db),
                f"{task}_qImages.npy": np.asarray(queries),
                f"{task}_gt.npy": gt,
            }
            for name, array in arrays.items():
                np.save(output / name, array, allow_pickle=False)
            manifest = {
                "dataset": task,
                "protocol_id": f"sues200_official_{direction}_{height}"
                if height is not None
                else f"sues200_mixed_heights_{direction}",
                "protocol_source": f"{split['repository']}/blob/{split['commit']}/script/split_datasets.py",
                "official_source": split,
                "split_sha256": sha256(SPLIT_FILE),
                "height": height,
                "direction": direction,
                "gallery_scope": "all_200_scenes",
                "query_scope": "80_test_scenes",
                "num_references": len(db),
                "num_queries": len(queries),
                "positives_per_query": int(gt.shape[1]),
                "reported_metric": "CMC@1, CMC@5",
                "generation_command": "python scripts/data/create_sues200_metadata.py",
                "files": {name: sha256(output / name) for name in arrays},
            }
            (output / f"{task}_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            manifests.append(manifest)
            print(
                f"{task}: gallery={len(db)}, query={len(queries)}, positives/query={gt.shape[1]}"
            )
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/sues200")
    parser.add_argument(
        "--metadata-root", help="Protocol output directory (default: --root)"
    )
    args = parser.parse_args()
    build_metadata(args.root, args.metadata_root)
