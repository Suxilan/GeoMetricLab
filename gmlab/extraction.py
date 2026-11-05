"""Public precision-controlled feature extraction with shared single/multi-rank publication."""

import importlib.metadata
import json
import os
import random
from functools import partial
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    SequentialSampler,
    default_collate,
)

from .artifacts import ArtifactStore, identity_key
from .utils.io import canonical_json
from .datasets.utils.feature_inputs import (
    dataset_preprocessing,
    dataset_samples,
)
from .models.identity import model_digest
from .extraction_identity import (
    _feature_calculation,
    _model_modules,
    _stream_calculation,
    validate_feature_source as validate_feature_source,
)
from .extraction_reuse import _reuse_rows
from .utils.precision import (
    resolve_inference_precision,
    inference_precision_context,
    inference_precision_record,
)


def _validate_loader(loader, *, explicit_adapter=False):
    if type(loader) is not DataLoader:
        raise TypeError("Extraction requires a native DataLoader for row selection")
    if not (
        callable(getattr(loader.dataset, "feature_samples", None))
        and hasattr(loader.dataset, "feature_preprocessing")
    ):
        raise ValueError(
            "Input identity requires feature_samples/feature_preprocessing"
        )
    if (
        type(loader.sampler) is not SequentialSampler
        or loader.drop_last
        or type(loader.batch_sampler) is not BatchSampler
        or loader.batch_size is None
        or (loader.collate_fn is not default_collate and not explicit_adapter)
    ):
        raise ValueError(
            "Global extraction requires a standard sequential loader/collate"
        )


def _numeric_identity(device, loader, seed, policy):
    numeric = {
        "backend": device.type,
        **inference_precision_record(device, policy),
        "default_dtype": str(torch.get_default_dtype()),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seed": seed,
        "batch_size": loader.batch_size,
        "workers": loader.num_workers,
        "loader": {
            "multiprocessing_context": (
                loader.multiprocessing_context.get_start_method()
                if loader.multiprocessing_context is not None
                else None
            ),
            "timeout": loader.timeout,
            "prefetch_factor": loader.prefetch_factor,
            "persistent_workers": loader.persistent_workers,
            "pin_memory": loader.pin_memory,
            "pin_memory_device": loader.pin_memory_device,
            "in_order": getattr(loader, "in_order", None),
            "row_selection": "exact_unpadded_rank_rows",
        },
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn": torch.backends.mkldnn.enabled,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        numeric.update(
            gpu=props.name,
            capability=[props.major, props.minor],
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
        )
    return numeric


@contextmanager
def _seeded(seed, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def extract_global_features(
    *,
    model: torch.nn.Module,
    model_config: Mapping[str, Any],
    loader: DataLoader,
    input_root: str | Path | None,
    data_name: str,
    method_name: str,
    embedding_adapter: Callable[..., Any] | None = None,
    batch_adapter: Callable[..., Any] | None = None,
    store: ArtifactStore | None = None,
    output: str | Path | None = None,
    device: str | torch.device = "cpu",
    storage_dtype: str = "float32",
    precision: str | int | None = None,
    matmul_tf32: bool | None = None,
    cudnn_tf32: bool | None = None,
    output_space: Mapping[str, Any] | None = None,
    parents: Sequence[Mapping[str, Any]] = (),
    seed: int = 0,
    realization: str = "deterministic_seed_v1",
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    algorithm_version: str = "global_fp32_model_adapter_v1",
    chunk_rows: int = 64,
) -> dict[str, Any]:
    """Extract one global [N,D] stream and return its publication record.

    Parameters and ownership follow extract_features, with no named streams:
    embedding_adapter selects finite [B,D] model output in input-row order.
    Choose exactly one of store or output (an H5 path); input_root scopes reader
    locators and may be None for readers with independent declarations.

    Return the global entry of extract_features(...)["streams"], including paths,
    identity/checksum, shape/dtype, hit/reuse counts and execution/timing records.
    No H5 handle or Tensor is returned. See extract_features and the artifact
    README for the complete record fields, precision and recovery contract.
    """
    return extract_features(
        model=model,
        model_config=model_config,
        loader=loader,
        input_root=input_root,
        data_name=data_name,
        method_name=method_name,
        embedding_adapter=embedding_adapter,
        batch_adapter=batch_adapter,
        store=store,
        output=output,
        device=device,
        storage_dtype=storage_dtype,
        precision=precision,
        matmul_tf32=matmul_tf32,
        cudnn_tf32=cudnn_tf32,
        output_space=output_space,
        parents=parents,
        seed=seed,
        realization=realization,
        provenance=provenance,
        overwrite=overwrite,
        algorithm_version=algorithm_version,
        chunk_rows=chunk_rows,
    )["streams"]["global"]


def _validate_extraction_request(
    model, loader, store, output, stream_specs, output_space, device, batch_adapter
):
    """Validate local runtime choices before entering collective extraction."""
    from .models.outputs import validate_stream_specs

    if (store is None) == (output is None):
        raise ValueError("Choose store or explicit output")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an nn.Module")
    _validate_loader(loader, explicit_adapter=batch_adapter is not None)
    specs = None if stream_specs is None else validate_stream_specs(stream_specs)
    if specs is not None and output_space is not None:
        raise ValueError("Declare named output semantics through stream_specs")
    if output is not None and specs is not None:
        if not isinstance(output, dict) or set(output) != set(specs):
            raise ValueError("Named exports require output paths for every stream")
        directories = [Path(path).resolve().parent for path in output.values()]
        if len(set(directories)) != len(directories):
            raise ValueError("Each exported stream needs its own sidecar directory")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU or CUDA devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no fallback")
        device = torch.device(
            "cuda",
            torch.cuda.current_device() if device.index is None else device.index,
        )
    return specs, device


def _describe_extraction_inputs(
    model,
    loader,
    *,
    model_config,
    policy,
    data_name,
    method_name,
    input_root,
    embedding_adapter,
    batch_adapter,
    output_space,
    storage_dtype,
    parents,
    seed,
    realization,
    algorithm_version,
):
    """Snapshot source rows and the numerical calculation before production."""
    if any(
        torch.is_tensor(value)
        and value.is_floating_point()
        and value.dtype != torch.float32
        for value in model.state_dict().values()
    ):
        raise ValueError("Model parameters and buffers must be FP32")
    # Freeze caller-owned nested declarations: later in-place changes
    # must not rewrite the baseline used by source validation.
    samples = dataset_samples(loader.dataset, source=data_name, input_root=input_root)
    samples = json.loads(canonical_json(samples))
    calculation = _feature_calculation(
        model=model,
        model_config=model_config,
        compute=policy,
        preprocessing=dataset_preprocessing(loader.dataset),
        embedding_adapter=embedding_adapter,
        batch_adapter=batch_adapter,
        output_space=output_space,
        storage_dtype=storage_dtype,
        parents=parents,
        seed=seed,
        realization=realization,
        algorithm_version=algorithm_version,
    )
    calculation = json.loads(canonical_json(calculation))
    identity = {
        "schema_version": 4,
        "kind": "global_features",
        "source": data_name,
        "selection": method_name,
        "calculation": {
            **calculation,
            "samples_sha256": identity_key(
                {
                    "rows": [
                        {k: v for k, v in sample.items() if k != "locator"}
                        for sample in samples
                    ]
                }
            ),
            "num_samples": len(samples),
        },
    }
    return identity, samples


def _verify_extraction_inputs(
    model, dataset, model_config, identity, samples, source, input_root
):
    """Reject changed sources before certifying private feature rows."""
    if identity_key(dataset_preprocessing(dataset)) != identity_key(
        identity["calculation"]["preprocessing"]
    ):
        raise ValueError("Input preprocessing changed during extraction")
    if identity_key(
        dataset_samples(dataset, source=source, input_root=input_root)
    ) != identity_key(samples):
        raise ValueError("Input changed during extraction")
    if model_digest(model) != identity["calculation"]["state_sha256"]:
        raise ValueError("Model state changed during extraction")
    if identity_key(_model_modules(model)) != identity_key(
        identity["calculation"]["model_modules"]
    ):
        raise ValueError("Model calculation changed during extraction")
    if identity_key(model_config) != identity_key(identity["calculation"]["model"]):
        raise ValueError("Model calculation declaration changed during extraction")


def extract_features(
    *,
    model: torch.nn.Module,
    model_config: Mapping[str, Any],
    loader: DataLoader,
    input_root: str | Path | None,
    data_name: str,
    method_name: str,
    embedding_adapter: Callable[..., Any] | None = None,
    batch_adapter: Callable[..., Any] | None = None,
    stream_specs: Mapping[str, Mapping[str, Any]] | None = None,
    store: ArtifactStore | None = None,
    output: str | Path | dict[str, str | Path] | None = None,
    device: str | torch.device = "cpu",
    storage_dtype: str = "float32",
    precision: str | int | None = None,
    matmul_tf32: bool | None = None,
    cudnn_tf32: bool | None = None,
    output_space: Mapping[str, Any] | None = None,
    parents: Sequence[Mapping[str, Any]] = (),
    seed: int = 0,
    realization: str = "deterministic_seed_v1",
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    algorithm_version: str = "global_fp32_model_adapter_v1",
    chunk_rows: int = 64,
) -> dict[str, Any]:
    """Produce verified global/spatial streams through one publication lifecycle.

    Inputs:
        model/model_config: caller-owned FP32 nn.Module and JSON-serializable
            calculation declaration, excluding weight locators. Eval output must
            be independent of other rows in the batch. Parameters are not updated.
        loader: native sequential DataLoader, no dropped tail/custom sampler.
            Its dataset declares feature_samples and feature_preprocessing.
            Floating model inputs must be FP32 before the configured autocast.
        input_root/data_name/method_name: locator scope, fixed data-version name
            and browsing method name. Dataset declarations define stable rows.
        batch_adapter: optional callable(batch) -> (model_inputs, batch_info).
            batch_info contains int64 index [B] and num_images; custom collation
            requires this adapter. Default image_batch consumes image/index.
        embedding_adapter: optional callable(model_output, batch_info) selecting
            outputs, without inventing input indices. None passes model output through.
        stream_specs: optional name -> declaration with layout global/spatial,
            fixed shape [D]/[C,H,W] and output semantics (models.outputs).
            Every named output has B rows aligned to the same original indices.
            Omit for a single global [B,D] stream described by output_space.
        store/output: exactly one destination. Cache uses ArtifactStore; export
            uses one H5 path or, for named streams, a dict of name -> H5 path in
            distinct directories. overwrite only permits explicit export replacement.
        precision/matmul_tf32/cudnn_tf32: shared inference policy; FP32 default,
            explicit AMP optional. storage_dtype independently selects float16/32/64.
        parents: verified artifact references with full keys/payload checksums.
        seed/realization/algorithm_version: deterministic realization and producer
            version included in identity. provenance is execution metadata only.
        chunk_rows: positive transfer/reuse chunk limit; local forward size uses
            loader.batch_size. Neither changes the batch-independent calculation.

    Returns:
        {"streams": {name: record}, "execution": stats}; the default name is global.
        Each record has schema_version, preprocessing, output_space, data_name,
        method_name; features/manifest/samples paths; artifact_key/calculation_key,
        features_sha256, portable reference; shape [N,D] or [N,C,H,W], dtype;
        cache_hit, inference_batches, reused_rows, reuse_sources; timing, device
        and execution. Paths identify closed files, not live handles. See the
        artifact README for reference/statistics details and public readers.

    State and failure boundaries:
        All initialized ranks must enter with identical contracts; rank zero owns
        locks/writers and publication. Each stream commits independently. The model
        is moved to device and left there; every module's training flag, scoped
        precision flags and global Python/NumPy/Torch RNG in the seeded scope
        are restored. The caller retains ownership of loader, dataset and
        ArtifactStore; this API does not close or reconfigure them. Iteration may
        advance a caller-supplied DataLoader generator through native PyTorch
        behavior. Caller generator/sampler/dataset/worker state and internal
        cursors are not snapshotted or restored. ArtifactStore is used normally
        to publish or reuse cache artifacts under the documented transaction rules.
        Valid controlled failures may retain certified private staging; only a
        successful source check after safe preparation permits continuation.
        Changed/unknown sources invalidate staging, never published artifacts.
        Invalid contracts or collective failures raise ValueError/RuntimeError;
        filesystem/H5 errors propagate through the shared runtime. SIGKILL,
        transport failure, power loss and HDF5 repair are outside recovery scope.
    """
    from .extraction_runtime import produce_feature_streams
    from .utils.distributed import check_rank_consistency
    from .datasets.utils.samples import image_batch

    error, specs = None, None
    try:
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive integer")
        for name, adapter in (
            ("batch_adapter", batch_adapter),
            ("embedding_adapter", embedding_adapter),
        ):
            if adapter is not None and not callable(adapter):
                raise TypeError(f"{name} must be callable or None")
        specs, device = _validate_extraction_request(
            model,
            loader,
            store,
            output,
            stream_specs,
            output_space,
            device,
            batch_adapter,
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if np.dtype(storage_dtype) not in (
            np.dtype("float16"),
            np.dtype("float32"),
            np.dtype("float64"),
        ):
            raise ValueError("Feature storage supports float16/32/64")
        policy = resolve_inference_precision(
            device,
            precision,
            matmul_tf32=matmul_tf32,
            cudnn_tf32=cudnn_tf32,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    check_rank_consistency(error)
    training = {module: module.training for module in model.modules()}
    try:
        model.to(device).eval()
        error, identity, samples = None, None, None
        try:
            identity, samples = _describe_extraction_inputs(
                model,
                loader,
                model_config=model_config,
                policy=policy,
                data_name=data_name,
                method_name=method_name,
                input_root=input_root,
                embedding_adapter=embedding_adapter,
                batch_adapter=batch_adapter,
                output_space=output_space,
                storage_dtype=storage_dtype,
                parents=parents,
                seed=seed,
                realization=realization,
                algorithm_version=algorithm_version,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        check_rank_consistency(
            error,
            {
                "identity": identity,
                "batch_size": loader.batch_size,
                "chunk_rows": chunk_rows,
                "specs": specs,
            },
        )
        active_store = store or ArtifactStore()
        if specs is None:
            identities, outputs = {"global": identity}, {"global": output}
        else:
            identities, outputs = {}, output or {}
            for name, spec in specs.items():
                calculation = _stream_calculation(identity["calculation"], name, spec)
                identities[name] = {
                    **identity,
                    "kind": (
                        "global_features"
                        if spec["layout"] == "global"
                        else "spatial_features"
                    ),
                    "calculation": calculation,
                }

        with (
            _seeded(seed, device),
            torch.inference_mode(),
            inference_precision_context(device, policy),
        ):
            return produce_feature_streams(
                model=model,
                loader=loader,
                identities=identities,
                samples=samples,
                specs=specs,
                store=active_store,
                outputs=outputs,
                device=device,
                embedding_adapter=embedding_adapter,
                batch_adapter=image_batch if batch_adapter is None else batch_adapter,
                storage_dtype=storage_dtype,
                overwrite=overwrite,
                provenance=provenance,
                reuse_rows=lambda *args: _reuse_rows(*args, chunk_rows=chunk_rows),
                verify_inputs=partial(
                    _verify_extraction_inputs,
                    model,
                    loader.dataset,
                    model_config,
                    identity,
                    samples,
                    data_name,
                    input_root,
                ),
                execution_context={
                    "numeric": _numeric_identity(device, loader, seed, policy),
                    "dependencies": {
                        name: importlib.metadata.version(name)
                        for name in ("torch", "torchvision", "Pillow", "numpy", "h5py")
                    },
                },
                chunk_rows=chunk_rows,
            )
    finally:
        for module, flag in training.items():
            module.training = flag


def predict_extraction(*, model, loader, trainer, **options):
    """Extract bounded feature streams using native Lightning prediction setup.

    The producer and returned artifact records match ``extract_features``.
    Trainer precision is inherited; conflicting explicit precision is rejected.
    """
    from .feature_prediction import predict_extraction as launch_prediction

    return launch_prediction(
        producer=extract_features,
        model=model,
        loader=loader,
        trainer=trainer,
        **options,
    )


def run_extraction(
    config: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
    batch_size: int | None = None,
    num_workers: int | None = None,
    loader_options: Mapping[str, Any] | None = None,
    cache_root: str | Path | None = None,
    output: str | Path | dict[str, str | Path] | None = None,
    overwrite: bool = False,
    devices: int | Sequence[int] | str | None = None,
    strategy: Any = "auto",
    trainer: Any = None,
    precision: str | int | None = None,
    matmul_tf32: bool | None = None,
    cudnn_tf32: bool | None = None,
    storage_dtype: str | None = None,
) -> dict[str, Any]:
    """Construct and run the shared producer from an extraction recipe.

    config requires model, data, transform, output and method. data contains
    dataset (component spec), root (input scope), name (fixed data version), and
    optional loader/transforms. transform may be null; output describes feature
    semantics, while this function's output argument chooses export H5 paths.
    Optional recipe keys: pipeline (or adapter/input_adapter), streams, initialization, seed,
    provenance, precision, matmul_tf32, cudnn_tf32, storage_dtype. Adapters accept
    callables/import paths. Model factories may supply resolved feature_config.

    loader_options override data.loader; explicit batch_size/num_workers override
    both, otherwise defaulting to 32/4. Numeric transforms must be declared by the
    reader, not hidden in collate/hooks. cache_root is used only in cache mode.
    Runtime precision arguments must agree with explicit recipe values.

    With trainer or devices, use Lightning's predict_extraction launcher; an
    injected Trainer is borrowed. strategy applies only when creating a Trainer.
    Otherwise call extract_features directly on device. Spawn requires importable,
    picklable models/adapters. Unlike extract_features, construction seeds Torch
    globally; it does not promise to restore caller RNG or constructed resources.

    Return one publication record without streams, otherwise the stream bundle
    documented by extract_features. Both add data_metadata from the dataset (or
    {}). Config/model/data/weight errors fail before production where possible;
    publication and recovery follow the same producer contract.
    """
    import inspect

    from .components import component_definition, resolve_component
    from .datasets.utils.loaders import build_loader

    allowed = {
        "model",
        "data",
        "transform",
        "output",
        "adapter",
        "input_adapter",
        "pipeline",
        "method",
        "seed",
        "provenance",
        "initialization",
        "streams",
        "precision",
        "matmul_tf32",
        "cudnn_tf32",
        "storage_dtype",
    }
    if config.keys() - allowed:
        raise ValueError(f"Unknown extraction options: {config.keys() - allowed}")
    for required in ("model", "data", "transform", "output", "method"):
        if required not in config:
            raise ValueError(f"Missing extraction section: {required}")
    if (batch_size is not None and batch_size < 1) or (
        num_workers is not None and num_workers < 0
    ):
        raise ValueError("Invalid loader settings")
    inference_options = {}
    for name, value in (
        ("precision", precision),
        ("matmul_tf32", matmul_tf32),
        ("cudnn_tf32", cudnn_tf32),
        ("storage_dtype", storage_dtype),
    ):
        if value is not None and config.get(name) is not None and value != config[name]:
            raise ValueError(f"Conflicting extraction {name} arguments")
        inference_options[name] = config.get(name) if value is None else value
    if trainer is None:
        resolve_inference_precision(
            device,
            inference_options["precision"],
            matmul_tf32=inference_options["matmul_tf32"],
            cudnn_tf32=inference_options["cudnn_tf32"],
        )
    inference_options["storage_dtype"] = inference_options["storage_dtype"] or "float32"
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    data = config["data"]
    options = {**data.get("loader", {}), **(loader_options or {})}
    for name, value, default in (
        ("batch_size", batch_size, 32),
        ("num_workers", num_workers, 4),
    ):
        if value is not None:
            options[name] = value
        options.setdefault(name, default)
    options.setdefault("pin_memory", str(device).startswith("cuda"))
    constructor, dataset_arguments = component_definition(data["dataset"])
    parameters = inspect.signature(constructor).parameters
    root_argument = {}
    if "root" not in dataset_arguments and "root" in parameters:
        # Keep the existing root-based reader convenience without forcing an
        # unrelated keyword on readers declaring metadata_path or other inputs.
        root_argument["root"] = data["root"]
    dataset, loader = build_loader(
        {
            "dataset": data["dataset"],
            "transform": config["transform"],
            "loader": options,
        },
        data.get("transforms"),
        seed=seed,
        dataset_arguments=root_argument,
    )
    model = build_extraction_model(config)
    from .pipeline import resolve_pipeline_route

    pipeline = resolve_pipeline_route(config)
    if pipeline is not None:
        if config.get("adapter") is not None or config.get("input_adapter") is not None:
            raise ValueError(
                "Extraction pipeline cannot be mixed with adapter/input_adapter"
            )
        pipeline.validate_model(model, inference=True)
        pipeline.validate_dataset(dataset, model)
        if not config.get("streams"):
            pipeline.validate_inference()
    adapters = {}
    for name in ("adapter", "input_adapter"):
        value = config.get(name)
        value = resolve_component(value) if isinstance(value, str) else value
        if value is not None and not callable(value):
            raise TypeError(f"Extraction {name} must be a callable or callable path")
        adapters[name] = value
    if pipeline is not None:
        adapters = {
            "adapter": pipeline.select_streams
            if config.get("streams")
            else pipeline.select_inference,
            "input_adapter": pipeline.prepare_inference,
        }
    options = dict(
        model=model,
        model_config=getattr(model, "feature_config", config["model"]),
        loader=loader,
        input_root=getattr(dataset, "input_root", data["root"]),
        data_name=data["name"],
        method_name=config["method"],
        embedding_adapter=adapters["adapter"],
        batch_adapter=adapters["input_adapter"],
        output_space=None if config.get("streams") else config["output"],
        stream_specs=config.get("streams"),
        seed=seed,
        store=ArtifactStore(cache_root) if output is None else None,
        output=output,
        overwrite=overwrite,
        provenance=config.get("provenance"),
        **inference_options,
    )
    if trainer is not None or devices is not None:
        if trainer is None:
            import pytorch_lightning as pl

            trainer = pl.Trainer(
                accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                devices=devices,
                strategy=strategy,
                precision=inference_options["precision"] or "32-true",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
        result = predict_extraction(trainer=trainer, **options)
    else:
        result = extract_features(device=device, **options)
    if config.get("streams") is None:
        result = result["streams"]["global"]
    result["data_metadata"] = getattr(dataset, "metadata", {})
    return result


def build_extraction_model(config: Mapping[str, Any]) -> torch.nn.Module:
    """Build the explicit model recipe, then apply optional pretrained state.

    Weight locators and format conversions do not select architecture. Direct
    constructors record resolved defaults; factories may expose feature_config.
    """
    import inspect

    from .components import component_definition
    from .models.weights import load_pretrained

    constructor, arguments = component_definition(config["model"])
    model = constructor(**arguments)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Extraction model must be an nn.Module")
    if not hasattr(model, "feature_config") and inspect.isclass(constructor):
        bound = inspect.signature(constructor).bind(**arguments)
        bound.apply_defaults()
        model.feature_config = {
            "type": f"{constructor.__module__}.{constructor.__qualname__}",
            "options": dict(bound.arguments),
        }
    if config.get("initialization") is not None:
        load_pretrained(model, **config["initialization"])
    return model
