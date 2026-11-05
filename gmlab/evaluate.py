"""Independent evaluation/extraction; no training Dataset/loss/optimizer built."""

import argparse
import json
from pathlib import Path

import pytorch_lightning as pl

from .artifacts import file_digest
from .config import read_recipe
from .models.identity import model_digest, model_identity_record
from .models.weights import restore_model
from .pipeline import resolve_pipeline_route
from .utils.precision import trainer_inference_options
from .training.checkpoints import resolve_checkpoint
from .training.module import MetricModule
from .training.provenance import evaluation_provenance
from .training.recipes import (
    build_evaluation_loaders,
    build_loader,
    loader_batch_processor,
    build_model,
    evaluation_sources,
    resolve_adapter,
    resolve_embedding_adapter,
    source_config,
)


def _extract_evaluation_features(
    config,
    loader,
    module,
    trainer,
    *,
    split,
    checkpoint,
    checkpoint_sha256,
    selection,
    output,
):
    """Publish the selected source using configured artifacts or an indexed export."""
    dataset = loader.dataset
    if config.get("artifacts") is not None:
        from .training.extraction import extract_artifact

        artifact = extract_artifact(
            config,
            loader,
            module,
            trainer,
            provenance={
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "selection": selection,
            },
        )
    else:
        from .extraction import predict_extraction

        input_root = getattr(dataset, "input_root", None)
        if input_root is None and hasattr(dataset, "paths"):
            import os

            input_root = os.path.commonpath(
                [str(Path(path).absolute().parent) for path in dataset.paths]
            )
        result = predict_extraction(
            model=module.model,
            model_config=config["model"],
            loader=loader,
            trainer=trainer,
            input_root=input_root,
            data_name=split,
            method_name=type(module.model).__name__.lower(),
            embedding_adapter=module.inference_embedding_adapter,
            batch_adapter=module.inference_batch_adapter,
            output=output.with_suffix(".h5"),
            provenance={"checkpoint_sha256": checkpoint_sha256},
            **module.inference_options,
        )
        artifact = result["streams"]["global"]
    return artifact


def run_evaluation(
    config,
    *,
    split,
    checkpoint,
    output,
    selection="explicit",
    extract=False,
    source=None,
):
    """Execute one selected split and save a self-contained result record.

    Evaluation consumes dataset-level GT; extraction needs only stable indices.
    Config snapshots distinguish annotation/selection identity from calculation
    identity. Extraction records its feature artifact and provenance. Path-based
    image inputs use the public H5 producer; custom adapters use indexed tensors.
    Named sources run through ordered native Lightning loaders, with independent
    indices, GT, adapters, metrics and calculation provenance. ``source`` selects
    one named source; extraction requires it instead of merging feature files.
    Legacy single-source records retain their flat metrics/provenance fields.
    No train reader, criterion, optimizer or scheduler is constructed.
    """
    declared_sources = evaluation_sources(config["data"][split])
    named = None not in declared_sources
    if source is not None:
        if source not in declared_sources:
            raise ValueError(f"Unknown source {source!r} for data.{split}")
        declared_sources = {source: declared_sources[source]}
    if extract and named and source is None:
        raise ValueError("Extraction from named sources requires an explicit source")
    selected_config = dict(config)
    selected_config["data"] = dict(config["data"])
    if named:
        selected_config["data"][split] = {"sources": declared_sources}
    stage = "val" if split == "valid" else "test"
    model = build_model(config)
    if extract:
        selected_spec = next(iter(declared_sources.values()))
        config = source_config(config, split, selected_spec)
        dataset, loader = build_loader(selected_spec, config["data"].get("transforms"))
        datasets = {source: dataset}
        evaluation = {}
    else:
        datasets, loader, stage_evaluation = build_evaluation_loaders(
            selected_config, split, model=model
        )
        evaluation = {stage: stage_evaluation}
        dataset = next(iter(datasets.values()))
    pipeline = resolve_pipeline_route(config)
    if pipeline is not None and extract:
        pipeline.validate_model(model, inference=True)
        pipeline.validate_inference()
        for source_dataset in datasets.values():
            pipeline.validate_dataset(source_dataset, model)
    checkpoint = Path(checkpoint)
    output = Path(output)
    step = restore_model(model, checkpoint)
    checkpoint_sha256 = file_digest(checkpoint)
    _, embedding_adapter = resolve_embedding_adapter(config, inference=True)
    module = MetricModule(
        model,
        embedding_adapter=embedding_adapter,
        inference_batch_adapter=(
            pipeline.prepare_inference
            if pipeline is not None
            else resolve_adapter(config["adapters"]["inference_batch"])
            if "inference_batch" in config.get("adapters", {})
            else loader_batch_processor(loader, inference=True)
            if extract
            else resolve_adapter("image")
        ),
        evaluation=evaluation,
    )
    runtime = config.get("runtime", {})
    trainer_options = dict(
        runtime.get("evaluation_trainer", {"accelerator": "cpu", "devices": 1})
    )
    if trainer_options.get("precision") not in (None, 32, "32", "32-true"):
        raise ValueError(
            "Retrieval evaluation requires evaluation_trainer.precision=32-true"
        )
    trainer_options["precision"] = "32-true"
    trainer = pl.Trainer(
        **trainer_options,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    inference = dict(runtime.get("inference", {}))
    if inference.get("precision") not in (None, 32, "32", "32-true"):
        raise ValueError(
            "Retrieval evaluation requires runtime.inference.precision=32-true"
        )
    if inference.get("matmul_tf32") or inference.get("cudnn_tf32"):
        raise ValueError("Retrieval evaluation requires TF32 to be disabled")
    inference["precision"] = "32-true"
    inference["matmul_tf32"] = False
    inference["cudnn_tf32"] = False
    module.inference_options = trainer_inference_options(trainer, **inference)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = None
    if extract:
        artifact = _extract_evaluation_features(
            config,
            loader,
            module,
            trainer,
            split=split,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            selection=selection,
            output=output,
        )
    elif stage == "val":
        trainer.validate(module, loader, verbose=False)
    else:
        trainer.test(module, loader, verbose=False)
    identity = (
        {"sha256": model_digest(model), "unavailable_reason": None}
        if extract
        else model_identity_record(model)
    )
    record = {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "selection": selection,
        "checkpoint_sha256": checkpoint_sha256,
        "training_step": step,
        "weights_sha256": identity["sha256"],
        "model_identity": identity,
        "split": split,
    }
    provenance = {}
    result = None if extract else module.evaluation_results[stage]
    for name, source_dataset in datasets.items():
        local = source_config(config, split, declared_sources[name])
        local_adapter_path, _ = resolve_embedding_adapter(local, inference=True)
        execution = (
            artifact["execution"]["numeric"]
            if artifact is not None
            else result["sources"][name]["inference_execution"]
            if named
            else module.inference_execution
        )
        provenance[name] = evaluation_provenance(
            local,
            split,
            source_dataset,
            model,
            trainer,
            embedding_adapter=local_adapter_path,
            extract=extract,
            inference_execution=execution,
        )
        if named:
            provenance[name]["inference_execution"] = execution
    if named:
        record["sources"] = provenance
        if source is not None:
            record["source"] = source
    else:
        record.update(provenance[None])

    if artifact is not None:
        record.update(
            {key: value for key, value in artifact.items() if key != "schema_version"}
        )
    else:
        record["metrics"] = result["metrics"]
        if named:
            for name in datasets:
                record["sources"][name]["metrics"] = result["sources"][name]["metrics"]
        record["inference_execution"] = (
            {name: provenance[name]["inference_execution"] for name in datasets}
            if named
            else module.inference_execution
        )
    if trainer.is_global_zero:
        output.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    """Evaluate or extract one selected collection using explicit saved weights."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split", choices=["valid", "test", "inference"], required=True
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--run-dir")
    parser.add_argument("--weights", choices=["best", "last"])
    parser.add_argument("--precision", choices=["32-true"])
    parser.add_argument(
        "--matmul-tf32", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--cudnn-tf32", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument(
        "--source",
        help="Select a named evaluation source; required for named extraction",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint, args.run_dir, args.weights)
    config = read_recipe(args.config)
    if args.precision is not None:
        runtime = config.setdefault("runtime", {})
        runtime.setdefault("evaluation_trainer", {})["precision"] = args.precision
        runtime.setdefault("inference", {})["precision"] = args.precision
    for key in ("matmul_tf32", "cudnn_tf32"):
        value = getattr(args, key)
        if value is not None:
            config.setdefault("runtime", {}).setdefault("inference", {})[key] = value
    run_evaluation(
        config,
        split=args.split,
        checkpoint=checkpoint,
        output=args.output,
        selection=args.weights or "explicit",
        extract=args.extract,
        source=args.source,
    )


if __name__ == "__main__":
    main()
