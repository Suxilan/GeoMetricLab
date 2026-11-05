#!/usr/bin/env python3
"""Build DenseUAV bidirectional metadata from the released scene-ID split.

The archive provides query_drone and gallery_satellite. The reverse direction
uses test-scene satellite images as queries and all train/test drone images as
the gallery, following the author's dataset-construction code and split table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COMMIT = "b8de18751675fcc0a7a40938b4cd66549647d690"
REPOSITORY = "https://github.com/Dmmm1997/DenseUAV"
SOURCE_SHA256 = {
    "README.md": "f863ca93708e5064a3448fc0b98b7f9fc293e7a8d268f68fadcdf7589c6833b4",
    "test.py": "6c90e02b1c88d341b5c12ed6a0b2b1acd1d54ee0eb84dc636b0a0b91408dc5df",
    "evaluate_gpu.py": "7ea21034c4f3afcb1727d1298817c8646cd6b3a915018df7281fce24435538c5",
    "tool/dataset_preprocess/3-generate_format_testset.py": "2ab72e72631e61b72cf65bdbc094c33374e9478a2ee3c4ef81f048c884abc23d",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_view(
    root: Path, relative: str, images_per_scene: int
) -> dict[str, list[str]]:
    view = root / relative
    if not view.is_dir():
        raise FileNotFoundError(f"Missing DenseUAV view: {view}")
    scenes = {}
    for directory in sorted(p for p in view.iterdir() if p.is_dir()):
        if len(directory.name) != 6 or not directory.name.isdecimal():
            raise ValueError(f"Invalid DenseUAV scene ID: {directory}")
        images = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) != images_per_scene:
            raise ValueError(
                f"Expected {images_per_scene} images in {directory}, found {len(images)}"
            )
        scenes[directory.name] = [p.relative_to(root).as_posix() for p in images]
    return scenes


def build_metadata(
    dataset_root: str | Path = "data/denseuav", metadata_root: str | Path | None = None
) -> list[dict]:
    root = Path(dataset_root)
    output = Path(metadata_root) if metadata_root is not None else root
    satellite = collect_view(root, "test/gallery_satellite", 6)
    test_drone = collect_view(root, "test/query_drone", 3)
    train_drone = collect_view(root, "train/drone", 3)
    train_satellite = collect_view(root, "train/satellite", 6)
    if (len(satellite), len(test_drone), len(train_drone)) != (3033, 777, 2256):
        raise ValueError(
            "DenseUAV release should have 3033 gallery, 777 test and 2256 train scene IDs"
        )
    if set(train_drone) & set(test_drone) or set(train_satellite) != set(train_drone):
        raise ValueError("DenseUAV training/test scene split is inconsistent")
    drone = {**train_drone, **test_drone}
    if set(drone) != set(satellite):
        raise ValueError("DenseUAV gallery must include every training and test scene")

    # Validate all required views before creating output files.
    output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for direction in ("drone2satellite", "satellite2drone"):
        task = f"denseuav_{direction}"
        db_view = satellite if direction == "drone2satellite" else drone
        q_view = (
            test_drone
            if direction == "drone2satellite"
            else {scene: satellite[scene] for scene in test_drone}
        )
        db = [p for scene in sorted(db_view) for p in db_view[scene]]
        queries = [p for scene in sorted(q_view) for p in q_view[scene]]
        indices, offset = {}, 0
        for scene in sorted(db_view):
            indices[scene] = np.arange(
                offset, offset + len(db_view[scene]), dtype=np.int64
            )
            offset += len(db_view[scene])
        gt = np.stack(
            [indices[scene] for scene in sorted(q_view) for _ in q_view[scene]]
        )
        arrays = {
            f"{task}_dbImages.npy": np.asarray(db),
            f"{task}_qImages.npy": np.asarray(queries),
            f"{task}_gt.npy": gt,
        }
        for name, array in arrays.items():
            np.save(output / name, array, allow_pickle=False)
        reconstructed = direction == "satellite2drone"
        manifest = {
            "dataset": task,
            "protocol_id": f"denseuav_{'reconstructed' if reconstructed else 'official'}_{direction}",
            "protocol_source": f"{REPOSITORY}/blob/{COMMIT}/tool/dataset_preprocess/3-generate_format_testset.py",
            "official_repository": REPOSITORY,
            "official_commit": COMMIT,
            "source_sha256": SOURCE_SHA256,
            "direction": direction,
            "heights": [80, 90, 100],
            "gallery_scope": "all_3033_scenes",
            "query_scope": "777_test_scenes",
            "train_ids": sorted(train_drone),
            "test_ids": sorted(test_drone),
            "num_references": len(db),
            "num_queries": len(queries),
            "positives_per_query": int(gt.shape[1]),
            "reverse_views_reconstructed": reconstructed,
            "reported_metric": "CMC@1, CMC@5; positives match scene ID, without height/date filtering",
            "generation_command": "python scripts/data/create_denseuav_metadata.py",
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
    parser.add_argument("--root", default="data/denseuav")
    parser.add_argument(
        "--metadata-root", help="Protocol output directory (default: --root)"
    )
    args = parser.parse_args()
    build_metadata(args.root, args.metadata_root)
