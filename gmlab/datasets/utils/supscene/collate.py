"""Pad scene subgraphs while keeping image, identity and GT rows aligned."""

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import torch

from ..identities import integer_tensor
from ..relations import prepare_relation_supervision
from ..samples import validate_sample_row


def make_pad_collate(
    diag_weight: float | None = None,
    *,
    pos_th: float = 0.3,
    neg_th: float | None = None,
    exclusive_scenes: bool | None = None,
    include_pair_weights: bool = False,
    positive_gamma: float = 2.0,
) -> Callable:
    """Return a picklable collate callable for variable-size scene subgraphs.

    Images become [B,N,C,H,W], overlap/pair_mask become [B,N,N], and
    node_idx/node_mask become [B,N]. N is the largest subgraph in this batch.
    node_mask excludes padding; pair_mask identifies observed within-subgraph GT,
    including zero-overlap negatives. Cross-subgraph relations are not created.
    supervision names static GT fields; it contains no objective names or kwargs.

    diag_weight retains optional data-side pair_weight output for independent
    consumers. GeoMetricLoss's training adapter explicitly rejects pair_weight;
    its diagonal/self policy is defined by identities and masks instead.
    """
    collate = partial(
        _pad_subgraphs,
        diag_weight=diag_weight,
        pos_th=pos_th,
        neg_th=neg_th,
        exclusive_scenes=exclusive_scenes,
        include_pair_weights=include_pair_weights,
        positive_gamma=positive_gamma,
    )
    collate.batch_format = "subgraphs"
    collate.relation_options = {"pos_th": pos_th, "neg_th": neg_th}
    return collate


def _pad_subgraphs(
    batch: Sequence[Mapping[str, Any]],
    *,
    diag_weight: float | None,
    pos_th: float,
    neg_th: float | None,
    exclusive_scenes: bool | None,
    include_pair_weights: bool,
    positive_gamma: float,
) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty subgraph batch")
    declarations = {item.get("exclusive_scenes") for item in batch}
    if len(declarations) != 1:
        raise ValueError("Subgraphs disagree on their GeoID exclusivity contract")
    declared = declarations.pop()
    if declared is not None and type(declared) is not bool:
        raise TypeError("Dataset exclusivity must be boolean")
    if exclusive_scenes is None:
        exclusive_scenes = False if declared is None else declared
    elif declared is not None and exclusive_scenes != declared:
        raise ValueError("Collate exclusivity contradicts the Dataset contract")
    image_presence = [item["images"] is not None for item in batch]
    if any(image_presence) and not all(image_presence):
        raise ValueError("Subgraphs must consistently provide images or None")
    for item in batch:
        count = integer_tensor(item["num_nodes"], name="num_nodes", minimum=1)
        if count.ndim != 0:
            raise ValueError("num_nodes must be a positive scalar integer")
        count = int(count)
        if item["overlap"].shape != (count, count):
            raise ValueError("Subgraph overlap must be exactly [N,N]")
        if len(item["image_paths"]) != count:
            raise ValueError("image_paths must align with num_nodes")
        if item["images"] is not None and (
            item["images"].ndim != 4 or len(item["images"]) != count
        ):
            raise ValueError("Subgraph images must be [N,C,H,W]")
    if all(image_presence) and any(
        item["images"].shape[1:] != batch[0]["images"].shape[1:] for item in batch
    ):
        raise ValueError("Subgraph image shapes must agree for padding")
    batch_size = len(batch)
    max_nodes = max(item["num_nodes"] for item in batch)
    images = None
    if batch[0]["images"] is not None:
        image_shape = batch[0]["images"].shape[1:]
        images = batch[0]["images"].new_zeros(batch_size, max_nodes, *image_shape)
    overlap = torch.zeros(batch_size, max_nodes, max_nodes)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    node_indices = torch.full((batch_size, max_nodes), -1, dtype=torch.long)
    pair_mask = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.bool)
    pair_weight = None
    if diag_weight is not None:
        pair_weight = torch.ones(batch_size, max_nodes, max_nodes)
    scene_ids, scene_keys, image_paths = [], [], []
    sample_ids, modalities, row_positions = [], [], []
    metadata_fields = {"sample_ids", "modalities", "row_positions"}
    has_metadata = [bool(metadata_fields & item.keys()) for item in batch]
    if any(has_metadata) and not all(metadata_fields <= item.keys() for item in batch):
        raise ValueError("Selected-row metadata must be complete for every subgraph")
    for batch_index, item in enumerate(batch):
        node_count = item["num_nodes"]
        scene_ids.append(item["scene_id"])
        scene_keys.append(item.get("scene_key", item["scene_id"]))
        image_paths.append(item["image_paths"])
        node_ids = integer_tensor(item["node_idx"], name="node_idx", minimum=0)
        if node_ids.shape != (node_count,):
            raise ValueError("node_idx must align with num_nodes [N]")
        if any(has_metadata):
            positions = integer_tensor(
                item["row_positions"], name="row_positions", minimum=0
            )
            if not torch.equal(positions, torch.arange(node_count)):
                raise ValueError("row_positions must preserve selected-node order")
            if (
                len(item["sample_ids"]) != node_count
                or len(item["modalities"]) != node_count
            ):
                raise ValueError("Selected-row metadata must align with num_nodes")
            identities, names = [], []
            for position, node_id in enumerate(node_ids.tolist()):
                row = validate_sample_row(
                    {
                        "inputs": None,
                        "sample_id": item["sample_ids"][position],
                        "modality": item["modalities"][position],
                    }
                )
                if row["sample_id"] != (scene_keys[-1], node_id):
                    raise ValueError(
                        "sample_ids must match scene and selected node identity"
                    )
                identities.append(row["sample_id"])
                names.append(row["modality"])
            sample_ids.append(identities)
            modalities.append(names)
            row_positions.append(positions)
        node_indices[batch_index, :node_count] = node_ids
        overlap[batch_index, :node_count, :node_count] = item["overlap"]
        node_mask[batch_index, :node_count] = True
        pair_mask[batch_index, :node_count, :node_count] = True
        if images is not None:
            images[batch_index, :node_count] = item["images"]
        if pair_weight is not None:
            diagonal = torch.arange(node_count)
            pair_weight[batch_index, diagonal, diagonal] = float(diag_weight)
    result = {
        "scene_ids": scene_ids,
        "scene_keys": scene_keys,
        "node_idx": node_indices,
        "image_paths": image_paths,
        "overlap": overlap,
        "node_mask": node_mask,
        "pair_mask": pair_mask,
        "images": images,
    }
    if any(has_metadata):
        result.update(
            sample_ids=sample_ids,
            modalities=modalities,
            row_positions=row_positions,
        )
    if pair_weight is not None:
        result["pair_weight"] = pair_weight
    result["supervision"] = prepare_relation_supervision(
        overlap,
        node_mask=node_mask,
        pair_mask=pair_mask,
        scene_ids=scene_keys,
        sample_ids=node_indices,
        pos_th=pos_th,
        neg_th=neg_th,
        exclusive_scenes=exclusive_scenes,
        include_pair_weights=include_pair_weights,
        positive_gamma=positive_gamma,
    )
    sources = {item.get("gt_source") for item in batch}
    if len(sources) != 1:
        raise ValueError("One subgraph batch must share a GT source")
    source = sources.pop()
    if source is not None:
        supervision = result["supervision"]
        supervision["relations"] = supervision["relations"]._replace(
            source=(source, pos_th, neg_th)
        )
    return result
