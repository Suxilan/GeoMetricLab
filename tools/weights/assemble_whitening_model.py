#!/usr/bin/env python3
"""Preview selected feature caches, explicitly fit whitening, optionally export an encoder."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gmlab.artifacts import ArtifactStore, file_digest
from gmlab.config import read_recipe
from gmlab.datasets.utils.feature_io import (
    feature_collection,
    iter_feature_collection,
    read_global_features,
)
from gmlab.extraction import build_extraction_model, validate_feature_source
from gmlab.models.assembly import attach_whitening, prepare_whitening
from gmlab.models.whitening import fit_whitening
from gmlab.utils.io import atomic_json
from tools.weights.stream_selection import aligned_stream_collection


def _feature_selection(args):
    """Parse user selections only; public readers determine identity/compatibility."""
    settings = {}
    combination = "independent"
    if args.feature_list:
        document = yaml.safe_load(args.feature_list.read_text())
        if not isinstance(document, dict) or document.keys() - {
            "features",
            "whitening",
            "combination",
        }:
            raise ValueError(
                "Feature-list YAML requires features and optional whitening"
            )
        features = document.get("features")
        combination = document.get("combination", "independent")
        settings = document.get("whitening", {})
        if not isinstance(settings, dict) or settings.keys() - {
            "dimension",
            "eps",
            "chunk_rows",
        }:
            raise ValueError("Unknown whitening settings in feature list")
        base = args.feature_list.resolve().parent
    else:
        key = "record" if args.extraction_record else "path"
        features = [{key: value} for value in (args.extraction_record or args.features)]
        base = Path.cwd()
    if combination != "independent" and (
        not isinstance(combination, dict)
        or set(combination) != {"type", "streams"}
        or combination["type"] != "aligned_streams"
    ):
        raise ValueError("Unknown whitening combination strategy")
    if not isinstance(features, list) or not features:
        raise ValueError(
            "Select at least one existing feature record or manifest-backed file"
        )
    selected = []
    for index, item in enumerate(features):
        if not isinstance(item, dict) or item.keys() - {"record", "path", "stream"}:
            raise ValueError(
                f"features[{index}] requires record or path, plus optional stream"
            )
        if ("record" in item) == ("path" in item):
            raise ValueError(f"features[{index}]: choose exactly one record or path")
        key = "record" if "record" in item else "path"
        if not isinstance(item[key], (str, Path)) or not str(item[key]):
            raise ValueError(f"features[{index}].{key} must be a nonempty path")
        path = Path(item[key]).expanduser()
        if not path.is_absolute():
            path = base / path
        stream = item.get("stream", args.stream)
        if stream is not None and (not isinstance(stream, str) or not stream):
            raise ValueError(f"features[{index}].stream must be a nonempty name")
        selected.append({key: path.resolve(), **({"stream": stream} if stream else {})})
    dimension = (
        args.whitening_dim
        if args.whitening_dim is not None
        else settings.get("dimension")
    )
    eps = args.eps if args.eps is not None else settings.get("eps", 1e-4)
    chunk_rows = (
        args.chunk_rows
        if args.chunk_rows is not None
        else settings.get("chunk_rows", 256)
    )
    if type(dimension) is not int or dimension < 1:
        raise ValueError(
            "Supply a positive whitening dimension in YAML or --whitening-dim"
        )
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("Whitening eps must be finite and positive")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    return selected, dimension, float(eps), chunk_rows, combination


def _collect_features(selections, *, cache_root, combination="independent"):
    """Adapt explicit choices to the shared verified collection contract."""
    if isinstance(combination, dict):
        return aligned_stream_collection(
            selections, streams=combination["streams"], cache_root=cache_root
        )
    collection_options = {
        "expected_output": {"whitening": False, "normalization": "none"},
        "cache_root": cache_root,
    }
    for key, argument in (("record", "records"), ("path", "paths")):
        if all(key in item and not item.get("stream") for item in selections):
            return _read_collection(
                selections,
                **{argument: [item[key] for item in selections]},
                **collection_options,
            )
    records = []
    for item in selections:
        source = item.get("record", item.get("path"))
        read_options = {"record": source} if "record" in item else {"path": source}
        if item.get("stream"):
            read_options["stream"] = item["stream"]
        try:
            data = read_global_features(
                load_values=False, cache_root=cache_root, **read_options
            )
        except (ValueError, OSError, KeyError) as error:
            raise ValueError(f"Feature input {source}: {error}") from error
        records.append(data["record"])
    return _read_collection(selections, records=records, **collection_options)


def _read_collection(selections, **options):
    """Keep the public field/index diagnostic linked to the user's input choices."""
    try:
        return feature_collection(**options)
    except ValueError as error:
        inputs = "; ".join(
            f"{index}={item.get('record', item.get('path'))}"
            + (f" stream={item['stream']}" if item.get("stream") else "")
            for index, item in enumerate(selections)
        )
        raise ValueError(f"{error}; selected inputs: {inputs}") from error


def _selection_summary(collection, selections, *, dimension, eps):
    inputs = []
    for item, source in zip(collection["inputs"], selections):
        manifest = item["manifest"]
        inputs.append(
            {
                "input": str(source.get("record", source.get("path"))),
                "stream": source.get("stream"),
                "data": manifest["identity"]["source"],
                "artifact_key": manifest["artifact_key"],
                "rows": item["shape"][0],
                "dimension": item["shape"][1],
                "dtype": item["dtype"],
                "output": manifest["identity"]["calculation"].get("output"),
            }
        )
    return {
        "compatible": True,
        "combination": collection.get("combination", "independent"),
        "sampling": "all_rows",
        "inputs": inputs,
        "total_rows": collection["shape"][0],
        "feature_dimension": collection["shape"][1],
        "whitening_dimension": dimension,
        "eps": eps,
        "weighting": "all_selected_rows_equal",
        "can_fit": collection["shape"][0] >= 2
        and dimension <= min(collection["shape"]),
    }


def _recipe_sources(path):
    """List input YAML files under read_recipe's single-relative-base contract."""
    paths = set()
    while path:
        path = Path(path).expanduser().resolve()
        if path in paths:
            raise ValueError("Cyclic encoder recipe inputs")
        paths.add(path)
        parent = yaml.safe_load(path.read_text()).get("extends")
        path = path.parent / parent if parent else None
    return paths


def _validate_outputs(args, collection, selections, *, store, recipe=None):
    """Check payloads, sidecars and reserved storage paths before any write/fit.

    Auxiliary files live outside the managed cache root: only ArtifactStore may
    write there. This also protects the as-yet unpublished cache destination
    without duplicating the store's key or destination selection algorithms.
    """
    destinations = {}
    if args.out:
        destinations["affine"] = args.out.expanduser().resolve()
        destinations["affine manifest"] = (
            destinations["affine"].parent / "manifest.json"
        )
    if args.encoder_out:
        destinations["encoder"] = args.encoder_out.expanduser().resolve()
        destinations["encoder metadata"] = (
            args.encoder_out.with_suffix(args.encoder_out.suffix + ".json")
            .expanduser()
            .resolve()
        )
    if args.record_output:
        destinations["run record"] = args.record_output.expanduser().resolve()
    for name, path in destinations.items():
        if (name != "affine manifest" and path.name == "manifest.json") or any(
            part in {".export.lock", ".staging", ".locks"} for part in path.parts
        ):
            raise ValueError(f"Output {name} uses a reserved artifact path: {path}")
        for other_name, other in destinations.items():
            if name != other_name and (path == other or path in other.parents):
                raise ValueError(
                    f"Output conflict: {name} {path} and {other_name} {other}"
                )
        if name in {
            "encoder",
            "encoder metadata",
            "run record",
        } and path.is_relative_to(store.root):
            raise ValueError(
                f"Output {name} must be outside the managed cache root: {path}"
            )
    if args.out:
        parent = destinations["affine"].parent
        reserved = tuple(
            (parent / name).resolve() for name in (".export.lock", ".staging")
        )
        for name, path in destinations.items():
            if any(
                path == entry or entry in path.parents or path in entry.parents
                for entry in reserved
            ):
                raise ValueError(
                    f"Output {name} conflicts with reserved export path: {path}"
                )
    protected = [Path(item["path"]).resolve().parent for item in collection["inputs"]]
    sources = {
        Path(item.get("record", item.get("path"))).resolve() for item in selections
    }
    sources.update(
        path.resolve() for path in (args.feature_list, args.encoder_config) if path
    )
    if args.encoder_config:
        sources.update(_recipe_sources(args.encoder_config))
    initialization = (recipe or {}).get("initialization") or {}
    if initialization.get("path"):
        sources.add(Path(initialization["path"]).expanduser().resolve())
    if any(
        any(path == source or path in source.parents for source in sources)
        or any(
            path.is_relative_to(parent) or path in parent.parents
            for parent in protected
        )
        for path in destinations.values()
    ):
        raise ValueError(
            "Whitening outputs must not overwrite selected inputs or their artifact directories"
        )
    if args.out and any(
        source == entry
        or source.is_relative_to(entry)
        or any(
            entry == directory or entry.is_relative_to(directory)
            for directory in protected
        )
        for source in sources
        for entry in reserved
    ):
        raise ValueError("Export control paths must not overlap protected inputs")
    for name, path in destinations.items():
        if path.is_dir() or any(parent.is_file() for parent in path.parents):
            raise ValueError(f"Output {name} is not a usable file path: {path}")
        if name in {"encoder", "encoder metadata"} and path.exists():
            raise FileExistsError(path)


def _fit_transform(collection, *, dimension, eps, chunk_rows, store, output, overwrite):
    """Return a publication record and its verified, in-memory affine snapshot.

    Capture both under the existing transaction lock, on creation and cache hit.
    Explicit exports may be replaced after this function returns; assembly must
    consume these tensors instead of reopening the mutable publication path.
    """
    parents = [
        {
            "artifact_key": item["manifest"]["artifact_key"],
            "payloads": item["manifest"]["payloads"],
        }
        for item in collection["inputs"]
    ]
    identity = {
        "schema_version": 4,
        "kind": "whitening",
        "source": "feature_collection",
        "selection": "pca_whitening",
        "calculation": {
            "algorithm": "pca_sample_covariance_ridge_v1",
            "feature_space": collection["calculation"],
            "parents": parents,
            "dimension": dimension,
            "eps": eps,
            "weighting": "all_selected_rows_equal",
        },
    }
    transaction = (
        store.export(identity, output, overwrite=overwrite)
        if output
        else store.write(identity, resume=False)
    )
    with transaction as tx:
        hit = tx["hit"]
        filename = tx.get("filename", "whitening.pt")
        if not hit:
            weight, bias = fit_whitening(
                iter_feature_collection(collection, chunk_rows=chunk_rows),
                dimension,
                eps=eps,
            )
            _validate_affine_parameters(
                {"weight": weight, "bias": bias}, dimension, collection["shape"][1]
            )
            torch.save(
                {"weight": weight.cpu(), "bias": bias.cpu()}, tx["path"] / filename
            )
            path = store.commit(
                tx,
                [
                    {
                        "path": filename,
                        "format": "torch_affine_whitening_v1",
                        "shape": list(weight.shape),
                        "dtype": str(weight.dtype),
                    }
                ],
                parents=parents,
                metadata={
                    "samples": collection["shape"][0],
                    "producer": "fit_whitening",
                },
            )
        else:
            path = tx["path"]
        manifest = store.verify(path, identity=identity)
        payload = next(
            item for item in manifest["payloads"] if item["path"] == filename
        )
        if (
            payload.get("format") != "torch_affine_whitening_v1"
            or payload.get("shape") != [dimension, collection["shape"][1]]
            or payload.get("dtype") != str(torch.float32)
        ):
            raise ValueError(
                "Whitening payload declaration differs from the requested affine"
            )
        parameters = torch.load(path / filename, map_location="cpu", weights_only=True)
        _validate_affine_parameters(parameters, dimension, collection["shape"][1])
        result = {
            "whitening": str(path / filename),
            "manifest": str(path / "manifest.json"),
            "artifact_key": manifest["artifact_key"],
            "whitening_sha256": payload["sha256"],
            "input_artifact_keys": collection["artifact_keys"],
            "dimension": manifest["identity"]["calculation"]["dimension"],
            "eps": manifest["identity"]["calculation"]["eps"],
            "cache_hit": hit,
        }
        if path.is_relative_to(store.root):
            result["reference"] = store.reference(path)
    return result, parameters


def _validate_affine_parameters(parameters, dimension, input_dimension):
    """Require the saved FP32 affine consumed by the model assembly contract."""
    if not isinstance(parameters, dict) or set(parameters) != {"weight", "bias"}:
        raise ValueError("Expected whitening weight and bias tensors")
    for name, shape in (
        ("weight", (dimension, input_dimension)),
        ("bias", (dimension,)),
    ):
        value = parameters[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"Expected finite FP32 whitening {name} with shape {shape}"
            )


def _mount_calculations(calculation):
    """Name explicit output selections; fitting policy does not choose model code."""
    if "spaces" in calculation:
        return calculation["spaces"]
    output = calculation["output"]
    name = output.get("stream") or output.get("branch") or "descriptor"
    return {name: calculation}


def _prepare_encoder(recipe, collection, *, dimension, config):
    """Bind verified source precision, then prepare the common affine mounting."""
    model = build_extraction_model(recipe)
    bound_recipe = dict(recipe)
    calculations = _mount_calculations(collection["calculation"])
    try:
        for name, calculation in calculations.items():
            try:
                binding = validate_feature_source(
                    bound_recipe, calculation, model=model
                )
                bound_recipe = {**bound_recipe, **binding}
            except ValueError as error:
                raise ValueError(f"Stream {name}: {error}") from error
        reconstruction = prepare_whitening(
            bound_recipe,
            calculations,
            model=model,
            dimension=dimension,
            assignments={name: "whitening" for name in calculations},
            trainable=True,
        )
    except ValueError as error:
        raise ValueError(f"Encoder recipe {config}: {error}") from error
    return model, reconstruction


def _export_encoder(prepared, calculation, transform, parameters, output):
    model, reconstruction = prepared
    calculations = _mount_calculations(calculation)
    model = attach_whitening(
        model,
        calculations,
        {"whitening": parameters},
        assignments={name: "whitening" for name in calculations},
        trainable=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    atomic_json(
        output.with_suffix(output.suffix + ".json"),
        {
            "operation": "attach_whitening",
            "whitening_artifact_key": transform["artifact_key"],
            "whitening_sha256": transform["whitening_sha256"],
            "input_artifact_keys": transform["input_artifact_keys"],
            "whitening_dimension": parameters["weight"].shape[0],
            "output_sha256": file_digest(output),
            "reconstruction_recipe": reconstruction,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--feature-list",
        type=Path,
        help="YAML cache selection; paths relative to this file",
    )
    source.add_argument("--extraction-record", nargs="+", type=Path)
    source.add_argument(
        "--features", nargs="+", type=Path, help="Manifest-backed feature H5 files"
    )
    parser.add_argument("--stream", help="Default named stream for selected records")
    parser.add_argument("--whitening-dim", type=int)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--chunk-rows", type=int)
    parser.add_argument("--cache-root", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preview",
        "--check",
        action="store_true",
        help="Read-only selection/compatibility check (default)",
    )
    mode.add_argument(
        "--fit",
        action="store_true",
        help="Explicitly fit or reuse affine whitening parameters",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional explicit affine-parameter export; otherwise publish to shared cache",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an explicit affine export only",
    )
    parser.add_argument("--record-output", type=Path)
    parser.add_argument(
        "--encoder-config", "--config", dest="encoder_config", type=Path
    )
    parser.add_argument(
        "--encoder-out",
        type=Path,
        help="Optional full encoder state, separate from affine --out",
    )
    args = parser.parse_args(argv)
    for name in ("out", "encoder_out", "record_output"):
        if getattr(args, name):
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.encoder_out and not args.encoder_config:
        parser.error("--encoder-out requires --encoder-config")
    if args.fit and args.encoder_config and not args.encoder_out:
        parser.error(
            "--encoder-config requires --encoder-out for assembly; omit the recipe for affine-only fitting"
        )
    selections, dimension, eps, chunk_rows, combination = _feature_selection(args)
    collection = _collect_features(
        selections, cache_root=args.cache_root, combination=combination
    )
    store = ArtifactStore(args.cache_root)
    recipe = read_recipe(args.encoder_config) if args.encoder_config else None
    _validate_outputs(args, collection, selections, store=store, recipe=recipe)
    summary = _selection_summary(collection, selections, dimension=dimension, eps=eps)
    model = (
        _prepare_encoder(
            recipe, collection, dimension=dimension, config=args.encoder_config
        )
        if args.encoder_config
        else None
    )
    if not args.fit:
        print(json.dumps(summary, indent=2))
        return summary
    if not summary["can_fit"]:
        raise ValueError(
            "Insufficient rows or whitening dimension exceeds the selected feature shape"
        )
    result, parameters = _fit_transform(
        collection,
        dimension=dimension,
        eps=eps,
        chunk_rows=chunk_rows,
        store=store,
        output=args.out,
        overwrite=args.overwrite,
    )
    if args.encoder_out:
        _export_encoder(
            model, collection["calculation"], result, parameters, args.encoder_out
        )
        result["encoder"] = str(args.encoder_out)
    if args.record_output:
        atomic_json(args.record_output, result)
    print(json.dumps({**summary, "result": result}, indent=2))
    return result


if __name__ == "__main__":
    main()
