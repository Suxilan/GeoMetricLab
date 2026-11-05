"""Small, explicit batch/output adapters; no dataset dispatch in the module."""

from collections.abc import Mapping
from typing import Any

import torch
from pytorch_metric_learning.utils.distributed import DistributedLossWrapper

from ..datasets.utils.samples import image_batch as image_batch

# Result positions are model_inputs, loss_kwargs and batch_info. loss_kwargs is
# forwarded only to the objective. batch_info always supplies num_images and may
# carry node_mask/modality_ids for output layout; inference additionally supplies
# stable index [N]. These dictionaries do not contain Trainer or optimizer state.
AdaptedBatch = tuple[Any, dict[str, Any], dict[str, Any]]


def grouped_inference_batch(batch: Mapping[str, Any]) -> tuple[dict, dict]:
    """Adapt modality Tensor groups and aligned manifest indices for inference.

    inputs/positions follow grouped_named_batch; index is int64 [N] in original
    row order. No supervision or image-specific metadata is required.
    """
    from ..models.branches import grouped_input_order

    inputs = {"inputs": batch["inputs"], "positions": batch["positions"]}
    order = grouped_input_order(inputs, batch["inputs"])
    indices = batch["index"]
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.long
        or indices.shape != (len(order),)
        or (indices < 0).any()
    ):
        raise ValueError("Grouped inference requires nonnegative int64 indices [N]")
    return inputs, {"num_images": len(order), "index": indices}


def grouped_named_batch(batch: Mapping[str, Any]) -> AdaptedBatch:
    """Route heterogeneous Tensor groups with aligned, independent supervision.

    inputs maps arbitrary modality names to Tensor[Nm,...]; positions maps each
    group to int64 rows partitioning [0,N). Each supervision entry retains its
    own GT semantics and addresses that original row order. No image shape,
    class identity or modality pairing is inferred here.
    """
    from ..models.branches import grouped_input_order

    inputs = {"inputs": batch["inputs"], "positions": batch["positions"]}
    order = grouped_input_order(inputs, batch["inputs"])
    supervision = batch["supervision"]
    if not isinstance(supervision, Mapping):
        raise TypeError("supervision must be a named data mapping")
    return inputs, {"supervision": supervision}, {"num_images": len(order)}


def named_batch(batch: Mapping[str, Any]) -> AdaptedBatch:
    """Pass Tensor inputs [N,...] and a target-independent supervision mapping.

    Heterogeneous model-input layouts can keep their own batch adapter; this
    simple contract deliberately does not infer row counts from arbitrary trees.
    """
    inputs = batch["inputs"]
    supervision = batch["supervision"]
    if not torch.is_tensor(inputs) or inputs.ndim == 0:
        raise ValueError("named_batch inputs must be a Tensor [N,...]")
    if not isinstance(supervision, Mapping):
        raise TypeError("named_batch supervision must be a named data mapping")
    return inputs, {"supervision": supervision}, {"num_images": len(inputs)}


def named_embeddings(
    output: Mapping[str, torch.Tensor], batch_info: Mapping
) -> Mapping:
    """Keep named differentiable Tensor outputs [N,...] in model-input row order.

    Metric targets select descriptors [N,D]; auxiliary regularizers may select
    spatial/intermediate features. Retrieval uses a separate single-output adapter.
    No Tensor leaves are detached, reordered, normalized or automatically pooled.
    """
    if not isinstance(output, Mapping) or not output:
        raise ValueError("named_embeddings requires a nonempty model-output mapping")
    for name, value in output.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Model output names must be nonempty strings")
        if not torch.is_tensor(value) or value.ndim == 0:
            raise ValueError(f"Model output {name!r} must be a Tensor [N,...]")
        if len(value) != batch_info["num_images"]:
            raise ValueError(f"Model output {name!r} rows must align with model inputs")
    return output


def modality_image_batch(batch: Mapping) -> tuple[dict, dict]:
    """Pass grouped model inputs and stable inference row indices separately."""
    count = len(batch["modalities"])
    indices = batch["index"]
    if indices.ndim != 1 or len(indices) != count or indices.dtype != torch.long:
        raise ValueError("Grouped inference indices must be int64 [N]")
    inputs = {"images": batch["images"], "positions": batch["positions"]}
    metadata = {"modalities": batch["modalities"], "positions": batch["positions"]}
    if "sample_ids" in batch:
        metadata["sample_ids"] = batch["sample_ids"]
    return inputs, {"num_images": count, "index": indices, **metadata}


def modality_id_batch(batch: Mapping) -> AdaptedBatch:
    """Pass grouped images to the model and aligned global class IDs to loss."""
    labels = batch["labels"]
    if labels.ndim != 1 or len(labels) != len(batch["modalities"]):
        raise ValueError("Grouped labels must align with original image rows")
    inputs = {"images": batch["images"], "positions": batch["positions"]}
    metadata = {
        key: batch[key]
        for key in (
            "sample_ids",
            "item_indices",
            "item_positions",
            "modalities",
            "modality_ids",
            "modality_names",
            "positions",
            "row_metadata",
        )
        if key in batch
    }
    targets = (
        {"indices_tuple": batch["indices_tuple"]}
        if "indices_tuple" in batch
        else {"labels": labels}
    )
    return inputs, targets, {"num_images": len(labels), **metadata}


def cross_modal_id_batch(batch: Mapping) -> AdaptedBatch:
    """Select only cross-modality ID pairs, within each rank's local batch.

    Equal class IDs define positives; unequal IDs define negatives. This is an
    explicit candidate policy, not a dataset heuristic. Use MetricObjective with
    distributed=False (no cross-rank candidate gather) and no additional miner.
    Ordinary DDP parameter synchronization remains owned by Lightning.
    """
    model_inputs, loss_kwargs, batch_info = modality_id_batch(batch)
    from ..datasets.utils.samples import cross_modality_pairs

    indices = cross_modality_pairs(batch["labels"], batch["modalities"])
    return model_inputs, {"indices_tuple": indices}, batch_info


def id_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> AdaptedBatch:
    """Adapt (inputs [N,...], labels [N]) to model input, kwargs and counts."""
    inputs, labels = batch
    if labels.ndim != 1 or len(inputs) != len(labels):
        raise ValueError("ID labels must be [N], aligned with model inputs")
    loss_kwargs = {"labels": labels}
    batch_info = {"num_images": len(labels)}
    return inputs, loss_kwargs, batch_info


def stacked_id_batch(batch: Mapping) -> AdaptedBatch:
    """Consume shared stacked rows while retaining identity/layout metadata.

    images [N,...] and labels [N] share original collate row order. Physical
    sample_ids and Dataset item_indices remain separate from class labels;
    repeated exposures retain their rows rather than being deduplicated.
    """
    images, labels = batch["images"], batch["labels"]
    if not torch.is_tensor(images) or images.ndim == 0:
        raise ValueError("Stacked inputs must be a Tensor [N,...]")
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("Stacked labels must align with input rows [N]")
    info = {
        key: batch[key]
        for key in (
            "sample_ids",
            "item_indices",
            "item_positions",
            "modalities",
            "modality_ids",
            "modality_names",
            "positions",
            "row_metadata",
        )
    }
    return images, {"labels": labels}, {"num_images": len(images), **info}


def multiview_id_batch(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> AdaptedBatch:
    """Consume the ID collate's (images, repeated labels, modality IDs).

    The selected single-tower encoder treats modalities identically. Other
    models can receive modality IDs through a separately injected adapter.
    Labels denote classes, not physical-image identities; no relation GT exists.
    """
    images, labels, modalities = batch
    if modalities.shape != labels.shape:
        raise ValueError("Modality IDs must align with labels")
    model_inputs, loss_kwargs, batch_info = id_batch((images, labels))
    batch_info = {**batch_info, "modality_ids": modalities}
    return model_inputs, loss_kwargs, batch_info


def relation_batch(batch: Mapping[str, Any]) -> AdaptedBatch:
    """Encode valid images from padded SupScene [B,N,C,H,W] in row order.

    node_mask [B,N] excludes padding before the encoder (including BN).
    pair_mask [B,N,N] marks observed relations inside each subgraph, including
    zero-overlap known negatives; collate excludes self/duplicate identities.
    scene_keys are canonical source-scoped identities, node_idx [B,N] identifies
    physical images within a scene. Cross-scene negatives require the caller's
    explicit exclusivity contract; this adapter does not enable that policy.
    Optional per-group sample_ids/modalities/row_positions contain only real
    nodes; they are flattened into encoder row order without padding identities.
    pair_weight is not consumed: existing loss objectives own their weighting.
    """
    if "pair_weight" in batch:
        raise ValueError(
            "pair_weight has no GeoMetricLoss contract; use objective weights"
        )
    images = batch["images"]
    valid_nodes = batch["node_mask"]
    if images is None or images.ndim != 5 or images.shape[:2] != valid_nodes.shape:
        raise ValueError("Expected images [B,N,C,H,W] and node_mask [B,N]")
    if valid_nodes.dtype != torch.bool or not valid_nodes.any():
        raise ValueError("node_mask must be boolean with at least one valid image")
    supervision = batch["supervision"]
    if not isinstance(supervision, Mapping):
        raise TypeError("SupScene supervision must be a named data mapping")
    loss_kwargs = {"supervision": supervision}
    model_inputs = images[valid_nodes]
    batch_info = {"node_mask": valid_nodes, "num_images": int(valid_nodes.sum())}
    metadata_fields = ("sample_ids", "modalities", "row_positions")
    present = [name in batch for name in metadata_fields]
    if any(present):
        if not all(present):
            raise ValueError(
                "Subgraph row metadata must include identity, modality and position"
            )
        counts = valid_nodes.sum(dim=1).tolist()
        for name in metadata_fields:
            groups = batch[name]
            if len(groups) != len(counts) or any(
                len(group) != count for group, count in zip(groups, counts)
            ):
                raise ValueError(f"{name} must align with valid subgraph nodes")
        batch_info["sample_ids"] = [
            identity for group in batch["sample_ids"] for identity in group
        ]
        batch_info["modalities"] = [
            name for group in batch["modalities"] for name in group
        ]
        batch_info["row_positions"] = torch.cat(batch["row_positions"])
    return model_inputs, loss_kwargs, batch_info


def tensor_embeddings(output: torch.Tensor, batch_info: Mapping) -> torch.Tensor:
    """Native models return [N,D]; never guess the meaning of extra axes."""
    if not torch.is_tensor(output) or output.ndim != 2:
        raise ValueError("Expected embeddings [N,D]; inject an explicit output adapter")
    if len(output) != batch_info["num_images"]:
        raise ValueError("Embeddings must align with valid input images")
    return output


def descriptor_embeddings(output: tuple | Mapping, batch_info: Mapping) -> torch.Tensor:
    """Read a descriptor Tensor, GeoEncoder tuple or explicit embeddings field.

    A dictionary model must explicitly provide the ``embeddings`` key.
    Auxiliary model outputs remain model-owned, and are not guessed as losses.
    """
    if torch.is_tensor(output):
        descriptors = output
    else:
        descriptors = output["embeddings"] if isinstance(output, Mapping) else output[0]
    descriptors = tensor_embeddings(descriptors, batch_info)
    return descriptors


def scalar_reports(
    output: torch.Tensor | Mapping, criterion: Any, world_size: int
) -> dict:
    """Detach scalar objective reports, preserving the backward Tensor.

    GeoMetricLoss already reports the global objective with its own backward
    scale. Only the native PML wrapper needs world-size reporting correction.
    Epoch values are valid-image-weighted means of batch reports, not a loss
    recomputed on all epoch candidates. All dictionary entries must be scalar;
    per-component entries are unweighted, main_loss/aux_loss are weighted sums.
    """
    reports = {"loss": output} if torch.is_tensor(output) else dict(output)
    if "loss" not in reports:
        raise ValueError("Criterion output must include loss")
    for key, value in reports.items():
        if not torch.is_tensor(value) or value.ndim != 0:
            raise ValueError(f"Criterion {key} must be a scalar Tensor")
    scale = world_size if isinstance(criterion, DistributedLossWrapper) else 1
    return {name: value.detach() / scale for name, value in reports.items()}
