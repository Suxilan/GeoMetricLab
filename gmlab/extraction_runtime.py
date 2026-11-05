"""Bounded distributed production for immutable feature streams.

All ranks enter together. Only rank zero owns transactions and feature writers.
Control collectives carry errors/indices, never feature arrays. Features travel
one rank and one chunk at a time to the publisher using native Tensor send/recv.
"""

import math
import resource
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_utilities.core.apply_func import apply_to_collection

from .datasets.utils.loaders import selected_loader as _selected_loader

from .datasets.utils.feature_io import (
    GlobalFeatureWriter,
    validate_feature_stream,
    write_samples,
    set_feature_source_verified,
)
from .datasets.utils.feature_description import validate_feature_output
from .models.outputs import align_output_streams
from .utils.distributed import (
    broadcast_object,
    check_rank_consistency,
    gather_objects,
    rank,
    world_size,
)


def _feature_record(
    store, path, identity, *, hit, batches, reused, sources, timings, device, execution
):
    manifest = store.verify(path, identity=identity, full=False)
    payload = next(
        item
        for item in manifest["payloads"]
        if item.get("format") in ("gmlab_global_h5_v1", "gmlab_spatial_h5_v1")
    )
    validate_feature_output(identity["calculation"]["output"], payload["shape"][1:])
    reference = {
        "artifact_key": manifest["artifact_key"],
        "payloads": manifest["payloads"],
    }
    if path.is_relative_to(store.root):
        reference.update(
            object=path.relative_to(store.root).as_posix(),
            feature_payload=payload["path"],
        )
    return {
        "schema_version": 1,
        "preprocessing": identity["calculation"]["preprocessing"],
        "output_space": identity["calculation"]["output"],
        "data_name": identity["source"],
        "method_name": identity["selection"],
        "features": str(path / payload["path"]),
        "manifest": str(path / "manifest.json"),
        "samples": str(path / "samples.jsonl"),
        "artifact_key": manifest["artifact_key"],
        "calculation_key": store.key(identity),
        "reference": reference,
        "features_sha256": payload["sha256"],
        "shape": payload["shape"],
        "dtype": payload["dtype"],
        "cache_hit": hit,
        "inference_batches": batches,
        "reused_rows": reused,
        "reuse_sources": sources,
        "timing": timings,
        "device": str(device),
        "execution": execution,
    }


def _move_inputs(value, device):
    """Move explicit model input trees without casting away the FP32 contract."""

    def validate(tensor):
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError("Loader must produce FP32 model inputs")
        return tensor

    return move_data_to_device(
        apply_to_collection(value, torch.Tensor, validate), device
    )


def _predict_rows(
    model, batch, pending_rows, *, device, specs, batch_adapter, embedding_adapter
):
    """Predict only unconsumed rank rows, allowing batch/arrival reordering."""
    if batch is None:
        if pending_rows:
            raise ValueError("Missing extraction input rows")
        return {}, None, 0.0
    inputs, info = batch_adapter(batch)
    indices = info["index"].detach().cpu()
    if (
        indices.ndim != 1
        or indices.dtype != torch.long
        or indices.unique().numel() != len(indices)
    ):
        raise ValueError("Input indices must be unique int64 [B]")
    if not set(indices.tolist()) <= pending_rows:
        raise ValueError("Dataset indices differ from selected input rows")
    if specs is not None:
        info = {**info, "output_streams": sorted(specs)}
    tick = time.perf_counter()
    output = model(_move_inputs(inputs, device))
    output = output if embedding_adapter is None else embedding_adapter(output, info)
    if specs is None:
        if (
            not torch.is_tensor(output)
            or not output.is_floating_point()
            or output.ndim != 2
            or len(output) != len(indices)
            or not torch.isfinite(output).all()
        ):
            raise ValueError("Adapter must return finite aligned Tensor [B,D]")
        values = {"global": output.detach()}
    else:
        values = {
            name: record["features"].detach()
            for name, record in align_output_streams(output, specs, indices).items()
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return values, indices, time.perf_counter() - tick


def _round_metadata(values, indices, *, seen, count, stats):
    """Agree row identities/layout before any feature Tensor is transferred."""
    local_bytes = sum(value.numel() * value.element_size() for value in values.values())
    metadata = gather_objects(
        {
            "indices": [] if indices is None else indices.tolist(),
            "shapes": {name: list(value.shape[1:]) for name, value in values.items()},
            "dtypes": {
                name: str(value.dtype).removeprefix("torch.")
                for name, value in values.items()
            },
            "output_bytes": local_bytes,
        }
    )
    failure = None
    if rank() == 0:
        shapes = [
            (item["shapes"], item["dtypes"]) for item in metadata if item["indices"]
        ]
        if shapes and any(shape != shapes[0] for shape in shapes):
            failure = "Output layout differs between ranks"
        for item in metadata:
            rows = item["indices"]
            if rows:
                stats["output_dtypes"].update(item["dtypes"])
            if rows and (min(rows) < 0 or max(rows) >= count or seen[rows].any()):
                failure = "Invalid/duplicate extracted input indices"
            if rows and min(rows) >= 0 and max(rows) < count:
                seen[rows] = True
            stats["max_rank_output_bytes"] = max(
                stats["max_rank_output_bytes"], item["output_bytes"]
            )
    check_rank_consistency(failure)
    return broadcast_object(metadata), local_bytes


def _write_rank_chunks(
    values, metadata, *, missing, publications, device, chunk_rows, stats
):
    """Drain each scheduled bounded transfer, then synchronize writer errors."""
    publisher = rank() == 0
    communication_device = (
        device
        if world_size() > 1 and dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    failure = None
    for source, item in enumerate(metadata):
        rows = item["indices"]
        for start in range(0, len(rows), chunk_rows):
            selected = np.asarray(rows[start : start + chunk_rows], dtype=np.int64)
            for name in sorted(missing):
                selected_mask = np.asarray([missing[name][i] for i in selected])
                if not selected_mask.any():
                    continue
                tensor = None
                if rank() == source:
                    tensor = (
                        values[name][start : start + len(selected)]
                        .to(device=communication_device)
                        .contiguous()
                    )
                if publisher and source != 0:
                    tensor = torch.empty(
                        (len(selected), *item["shapes"][name]),
                        device=communication_device,
                        dtype=getattr(torch, item["dtypes"][name]),
                    )
                    dist.recv(tensor, src=source)
                elif rank() == source and source != 0:
                    dist.send(tensor, dst=0)
                if publisher:
                    stats["max_transfer_bytes"] = max(
                        stats["max_transfer_bytes"],
                        tensor.numel() * tensor.element_size(),
                    )
                    if failure is None:
                        try:
                            publications[name].writer.write(
                                selected[selected_mask],
                                tensor.cpu()[torch.from_numpy(selected_mask)],
                            )
                        except Exception as exc:
                            # Keep receiving after failure so peers never remain
                            # blocked in a send while rank zero closes its writer.
                            failure = f"{type(exc).__name__}: {exc}"
    check_rank_consistency(failure)


@dataclass
class StreamPublication:
    """One stream's coupled transaction, initial hit, staging rows and reuse.

    Inference writers belong exclusively to the writer ExitStack. This record
    never closes resources. Recovery invalidation may reopen private payloads
    only after that stack exits, under the same transaction lock.
    """

    name: str
    identity: dict
    specification: dict | None
    transaction: dict
    cache_hit: bool
    writer: GlobalFeatureWriter | None
    missing: list[bool]
    reused_rows: int
    reuse_sources: list[str]
    reuse_chunk_bytes: int
    # True only after this attempt has durably marked pending and, if needed,
    # cleared historical rows. Current source agreement alone is insufficient.
    source_prepared: bool = False

    @property
    def feature_path(self) -> Path:
        return self.transaction["path"] / self.transaction.get(
            "filename", "features.h5"
        )


def _prepare_publications(
    *,
    identities,
    samples,
    specs,
    store,
    outputs,
    overwrite,
    storage_dtype,
    reuse_rows,
    chunk_rows,
    transactions,
    writers,
    publications,
):
    """Open rank-zero transactions/writers and capture one missing plan per stream."""

    def lock_order(name):
        destination = outputs.get(name)
        return (
            ("cache", store.key(identities[name]))
            if destination is None
            else ("export", str(Path(destination).resolve().parent))
        )

    for name in sorted(identities, key=lock_order):
        identity, destination = identities[name], outputs.get(name)
        context = (
            store.write(identity)
            if destination is None
            else store.export(identity, destination, overwrite=overwrite)
        )
        tx = transactions.enter_context(context)
        specification = None if specs is None else specs[name]
        state = StreamPublication(
            name=name,
            identity=identity,
            specification=specification,
            transaction=tx,
            cache_hit=tx["hit"],
            writer=None,
            missing=[False] * len(samples),
            reused_rows=0,
            reuse_sources=[],
            reuse_chunk_bytes=0,
        )
        # Track the transaction before writer/reuse setup can fail. Recovery
        # validation must also reach partially prepared streams.
        publications[name] = state
        if not tx["hit"]:
            write_samples(tx["path"] / "samples.jsonl", samples)
            writer = GlobalFeatureWriter(
                tx["path"] / tx.get("filename", "features.h5"),
                samples,
                dtype=storage_dtype,
                shape=None if specification is None else specification["shape"],
            )
            # This stack is the sole closer, before either commit or lock release.
            writers.callback(writer.close)
            state.writer = writer
            writer.begin_source_validation()
            # Only successful preparation proves historical rows were either
            # certified already or cleared before this attempt can produce.
            state.source_prepared = True
            state.reused_rows, state.reuse_sources = reuse_rows(
                store, identity, samples, writer
            )
            if state.reused_rows:
                values = writer.file["features"]
                state.reuse_chunk_bytes = (
                    chunk_rows * int(np.prod(values.shape[1:])) * values.dtype.itemsize
                )
            state.missing = (~writer.file["written"][:]).tolist()
    return publications


def _execute_feature_batches(
    *,
    model,
    loader,
    samples,
    specs,
    missing,
    publications,
    device,
    batch_adapter,
    embedding_adapter,
    chunk_rows,
    stats,
):
    """Run the common rank plan; synchronize failures before payload transfers."""
    needed = [
        i for i in range(len(samples)) if any(rows[i] for rows in missing.values())
    ]
    local_rows = needed[rank() :: world_size()]
    remaining = set(local_rows)
    rounds = math.ceil(math.ceil(len(needed) / world_size()) / loader.batch_size)
    iterator, error = None, None
    try:
        iterator = (
            iter(_selected_loader(loader, local_rows)) if local_rows else iter(())
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    check_rank_consistency(error)
    seen = np.zeros(len(samples), dtype=bool) if rank() == 0 else None
    batches, compute_seconds, max_output_bytes = 0, 0.0, 0
    prepared_at = time.perf_counter()
    for _ in range(rounds):
        values, indices, error = {}, None, None
        try:
            values, indices, elapsed = _predict_rows(
                model,
                next(iterator, None),
                remaining,
                device=device,
                specs=specs,
                batch_adapter=batch_adapter,
                embedding_adapter=embedding_adapter,
            )
            if indices is not None:
                remaining.difference_update(indices.tolist())
            compute_seconds += elapsed
            batches += int(indices is not None)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        check_rank_consistency(error)
        metadata, local_bytes = _round_metadata(
            values, indices, seen=seen, count=len(samples), stats=stats
        )
        max_output_bytes = max(max_output_bytes, local_bytes)
        _write_rank_chunks(
            values,
            metadata,
            missing=missing,
            publications=publications,
            device=device,
            chunk_rows=chunk_rows,
            stats=stats,
        )
    check_rank_consistency(
        "ValueError: Missing extraction input rows" if remaining else None
    )
    if rank() == 0:
        # A resumed staging file may need validation without inference/reuse.
        # Include its forthcoming finite-scan chunk in the feature-buffer bound.
        validation_bytes = []
        for state in publications.values():
            if state.writer is not None:
                values = state.writer.file["features"]
                validation_bytes.append(
                    min(len(values), chunk_rows)
                    * int(np.prod(values.shape[1:]))
                    * values.dtype.itemsize
                )
        stats["max_validation_chunk_bytes"] = max(validation_bytes, default=0)
    execution = gather_objects(
        {
            "batches": batches,
            "compute_seconds": compute_seconds,
            "selected_rows": len(local_rows),
            "max_output_bytes": max_output_bytes,
            "process_lifetime_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            * 1024,
        }
    )
    return execution, prepared_at


def _publish_feature_streams(
    *,
    publications,
    store,
    count,
    chunk_rows,
    provenance,
    execution_context,
    execution,
    stats,
    started,
    prepared_at,
    device,
):
    """Validate closed staging payloads and publish each immutable stream."""
    records = {}
    stats["max_reuse_chunk_bytes"] = max(
        (state.reuse_chunk_bytes for state in publications.values()), default=0
    )
    stats["rank0_feature_buffer_bound_bytes"] = max(
        stats["max_rank_output_bytes"] + 3 * stats["max_transfer_bytes"],
        3 * stats["max_reuse_chunk_bytes"],
        3 * stats["max_validation_chunk_bytes"],
    )
    stats["ranks"] = execution
    stats["rank0_process_lifetime_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    for name, state in sorted(publications.items()):
        if not state.cache_hit:
            schema = validate_feature_stream(
                state.feature_path,
                count,
                shape=(
                    None
                    if state.specification is None
                    else state.specification["shape"]
                ),
                chunk_rows=chunk_rows,
            )
            validate_feature_output(
                state.identity["calculation"]["output"], schema["shape"][1:]
            )
            store.commit(
                state.transaction,
                [schema, {"path": "samples.jsonl", "format": "jsonl"}],
                metadata={
                    "num_samples": count,
                    "reused_rows": state.reused_rows,
                    "reuse_sources": state.reuse_sources,
                    "producer": "gmlab.extraction.extract_features",
                    "origin": provenance or {},
                    "execution": {**execution_context, **stats},
                },
                parents=state.identity["calculation"]["parents"],
            )
        records[name] = _feature_record(
            store,
            state.transaction["path"],
            state.identity,
            hit=state.cache_hit,
            batches=sum(item["batches"] for item in execution),
            reused=state.reused_rows,
            sources=state.reuse_sources,
            timings={
                "preparation_seconds": prepared_at - started,
                "compute_seconds": max(item["compute_seconds"] for item in execution),
                "total_seconds": time.perf_counter() - started,
            },
            device=device,
            execution=stats,
        )
    return {"streams": records, "execution": stats}


def _record_source_verification(publications, *, verified):
    """Update private closed payloads on the publisher; share any IO failure."""
    error = None
    if rank() == 0:
        try:
            for state in publications.values():
                # commit changes tx.hit to True. Never mutate an immutable hit
                # or a stream already published successfully.
                if not state.transaction["hit"] and (
                    not verified or state.source_prepared
                ):
                    set_feature_source_verified(state.feature_path, verified=verified)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
    check_rank_consistency(error)


def _check_production_state(
    error, publications, verify_inputs, *, verify_on_success=False
):
    """Certify safely prepared rows after all ranks confirm the original source.

    Enter at synchronized boundaries after writers close and while transaction
    locks remain held. Failed preparation, iteration, inference, transfer and
    close paths use the same check; publication failures recheck before release.
    """
    failure = None
    try:
        check_rank_consistency(error)
    except Exception as exception:
        failure = exception
    if failure is None and not verify_on_success:
        return
    source_error = None
    try:
        verify_inputs()
    except Exception as exception:
        source_error = f"{type(exception).__name__}: {exception}"
    try:
        check_rank_consistency(source_error)
    except Exception as source_failure:
        _record_source_verification(publications, verified=False)
        raise source_failure from failure
    _record_source_verification(publications, verified=True)
    if failure is not None:
        raise failure


def produce_feature_streams(
    *,
    model,
    loader,
    identities,
    samples,
    specs,
    store,
    outputs,
    device,
    embedding_adapter,
    batch_adapter,
    storage_dtype,
    overwrite,
    provenance,
    reuse_rows,
    verify_inputs,
    execution_context,
    chunk_rows,
):
    """Prepare stream state, execute the rank plan, then publish closed payloads.

    Only rank zero owns stream resources. The inner ExitStack closes every H5
    exactly once, before publication. The outer stack holds transaction locks
    until publication/record broadcast ends, including exceptional exits.
    Feature buffers remain one model batch per rank plus bounded transfers;
    missing/seen metadata is O(N*streams). Process-loss recovery is not promised.
    """
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    started = time.perf_counter()
    publisher = rank() == 0
    publications, missing, error = {}, None, None
    stats = {
        "publisher_rank": 0,
        "world_size": world_size(),
        "chunk_rows": chunk_rows,
        "max_transfer_bytes": 0,
        "max_rank_output_bytes": 0,
        "output_dtypes": {},
        "numeric": execution_context["numeric"],
        "storage_dtype": str(np.dtype(storage_dtype)),
    }
    with ExitStack() as transactions:
        try:
            with ExitStack() as writers:
                if publisher:
                    try:
                        publications = _prepare_publications(
                            identities=identities,
                            samples=samples,
                            specs=specs,
                            store=store,
                            outputs=outputs,
                            overwrite=overwrite,
                            storage_dtype=storage_dtype,
                            reuse_rows=reuse_rows,
                            chunk_rows=chunk_rows,
                            transactions=transactions,
                            writers=writers,
                            publications=publications,
                        )
                        missing = {
                            name: state.missing for name, state in publications.items()
                        }
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                check_rank_consistency(error)
                missing = broadcast_object(missing)
                execution, prepared_at = _execute_feature_batches(
                    model=model,
                    loader=loader,
                    samples=samples,
                    specs=specs,
                    missing=missing,
                    publications=publications,
                    device=device,
                    batch_adapter=batch_adapter,
                    embedding_adapter=embedding_adapter,
                    chunk_rows=chunk_rows,
                    stats=stats,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _check_production_state(
            error, publications, verify_inputs, verify_on_success=True
        )
        # No H5 writer remains open while validation or publication runs.
        result, error = None, None
        if publisher:
            try:
                result = _publish_feature_streams(
                    publications=publications,
                    store=store,
                    count=len(samples),
                    chunk_rows=chunk_rows,
                    provenance=provenance,
                    execution_context=execution_context,
                    execution=execution,
                    stats=stats,
                    started=started,
                    prepared_at=prepared_at,
                    device=device,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        _check_production_state(error, publications, verify_inputs)
        return broadcast_object(result)
