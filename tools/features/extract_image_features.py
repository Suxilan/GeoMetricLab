#!/usr/bin/env python3
"""Extract image features using the shared gmlab producer and an explicit recipe."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gmlab.config import read_recipe
from gmlab.extraction import run_extraction
from gmlab.utils.io import atomic_json


def _devices(value):
    """Parse native Lightning device counts, index lists, -1 or auto."""
    if value == "auto":
        return value
    try:
        if "," in value:
            indices = [int(part) for part in value.split(",")]
            if min(indices) < 0 or len(set(indices)) != len(indices):
                raise ValueError
            return indices
        count = int(value)
        if count == 0 or count < -1:
            raise ValueError
        return count
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Use a positive count, unique device indices, -1 or auto"
        ) from None


def _publish_record(result, path):
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

    @rank_zero_only
    def publish():
        records = result["streams"].values() if "streams" in result else [result]
        payload_paths = {
            Path(record[key]).resolve()
            for record in records
            for key in ("features", "samples", "manifest")
        }
        if path.resolve() in payload_paths:
            raise ValueError("Run record must not overwrite an artifact")
        atomic_json(path, result)
        print(json.dumps(result, indent=2))

    publish()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path)
    parser.add_argument("--input-root", type=str)
    parser.add_argument("--data-name")
    parser.add_argument("--image-list")
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument("--image-size", type=int)
    geometry.add_argument(
        "--no-resize",
        action="store_true",
        help="Remove Resize only; preserve other preprocessing. Use batch size 1 for mixed image sizes.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--precision",
        choices=("32-true", "bf16-mixed", "16-mixed"),
        help="Public inference compute policy; default comes from YAML or FP32",
    )
    parser.add_argument(
        "--matmul-tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicit CUDA matrix-multiplication TF32 policy",
    )
    parser.add_argument(
        "--cudnn-tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicit CUDA convolution TF32 policy",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32", "float64"),
        help="Feature payload dtype, independent of compute precision",
    )
    parser.add_argument(
        "--devices", type=_devices, help="Native Lightning device count or index list"
    )
    parser.add_argument(
        "--strategy", default="auto", help="Native Lightning prediction strategy"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Override data.loader.batch_size (default 32)"
    )
    parser.add_argument(
        "--num-workers", type=int, help="Override data.loader.num_workers (default 4)"
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--record-output", type=Path, required=True)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path)
    output.add_argument(
        "--stream-output",
        action="append",
        metavar="NAME=H5",
        help="Explicit export per named stream; repeat for every stream",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.devices is None and args.strategy != "auto":
        parser.error("--strategy requires --devices")
    if args.devices is not None and args.device.startswith("cuda:"):
        parser.error(
            "Use --device cuda with --devices, or omit --devices for one cuda:INDEX"
        )
    destination = args.output
    if args.stream_output:
        destination = {}
        for item in args.stream_output:
            name, separator, path = item.partition("=")
            if not separator or not name or not path or name in destination:
                parser.error("--stream-output requires distinct NAME=H5 entries")
            destination[name] = Path(path)
    config = read_recipe(args.config)
    if args.data_config:
        config["data"] = read_recipe(args.data_config)
    if args.input_root:
        config["data"]["root"] = args.input_root
    if args.data_name:
        config["data"]["name"] = args.data_name
    if args.image_list:
        config["data"]["dataset"]["image_list"] = args.image_list
    if args.no_resize:
        if not isinstance(config["transform"], list):
            parser.error("--no-resize requires an explicit operation list")
        config["transform"] = [
            op for op in config["transform"] if op["type"].split(".")[-1] != "Resize"
        ]
    if args.image_size is not None:
        if args.image_size < 1:
            parser.error("--image-size must be positive")
        if not isinstance(config["transform"], list):
            parser.error("--image-size requires an explicit operation list")
        if not any(op["type"].split(".")[-1] == "Resize" for op in config["transform"]):
            parser.error("Recipe has no Resize to override; choose a resize recipe")
        for operation in config["transform"]:
            if operation["type"].split(".")[-1] == "Resize":
                operation["size"] = [args.image_size, args.image_size]
    if args.limit:
        config["data"]["dataset"]["limit"] = args.limit
    result = run_extraction(
        config,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_root=args.cache_root,
        output=destination,
        overwrite=args.overwrite,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        matmul_tf32=args.matmul_tf32,
        cudnn_tf32=args.cudnn_tf32,
        storage_dtype=args.storage_dtype,
    )
    _publish_record(result, args.record_output)
    return result


if __name__ == "__main__":
    main()
