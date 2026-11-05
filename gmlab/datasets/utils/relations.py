"""Static relation GT conversion, independent of descriptors and training."""

import hashlib
import math
from typing import NamedTuple

import torch
import numpy as np

from .identities import integer_tensor


class RelationSupervision(NamedTuple):
    """Observed flat [M,M] masks and identities for M valid model input rows.

    row_indices selects valid rows from the original flattened [B,N] layout.
    Equal known (scene, sample) identities are excluded. Different subgraphs of
    one scene have unknown relations. exclusive_scenes explicitly declares that
    distinct known scene identities imply negatives; its default is False.
    """

    positive: torch.Tensor
    negative: torch.Tensor
    known: torch.Tensor
    groups: torch.Tensor
    scene_ids: torch.Tensor
    sample_ids: torch.Tensor
    num_groups: int
    row_indices: torch.Tensor
    padded_shape: tuple[int, int]
    exclusive_scenes: bool = False
    source: tuple[str, float, float | None] | None = None


def stable_id(value: str) -> int:
    """Return a deterministic nonnegative source-scoped integer identity."""
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _identities(values, batch_size, nodes_per_group, valid_nodes, device, name):
    if values is None:
        return torch.full_like(valid_nodes, -1, dtype=torch.long)[valid_nodes]
    if isinstance(values, np.ndarray) and values.ndim == 1 and values.dtype.kind == "U":
        values = values.tolist()
    if isinstance(values, (list, tuple)) and any(isinstance(v, str) for v in values):
        values = [stable_id(v) if isinstance(v, str) else v for v in values]
    identities = integer_tensor(values, device=device, name=name).reshape(-1)
    if identities.numel() == batch_size:
        identities = identities.repeat_interleave(nodes_per_group)
    if identities.numel() != batch_size * nodes_per_group:
        raise ValueError(f"{name} must contain B or B*N identities")
    return identities[valid_nodes]


def prepare_relations(
    overlap: torch.Tensor | None = None,
    *,
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    pair_mask: torch.Tensor | None = None,
    node_mask: torch.Tensor | None = None,
    scene_ids=None,
    sample_ids=None,
    pos_th: float = 0.3,
    neg_th: float | None = None,
    exclusive_scenes: bool = False,
) -> RelationSupervision:
    """Convert static [B,N,N] GT into observed PML candidate masks.

    Scores >= pos_th are positive; scores below pos_th are negative unless
    neg_th declares a neutral interval. pair_mask marks observed entries: an
    observed zero is a negative, an unobserved entry stays unknown. node_mask
    removes padding in row-major order. No class labels are synthesized.

    Numeric identities must be signed-int64-representable integer data; floats
    (even integral floats) and booleans are rejected before casting or padding
    removal. Negative identities remain unknown. Flat string keys use stable_id;
    callers must namespace scene keys across data sources. A physical image is
    scoped by (known scene_id, known sample_id), not a sample_id alone.
    """
    if type(exclusive_scenes) is not bool:
        raise TypeError("exclusive_scenes must be bool")
    if not math.isfinite(pos_th) or (neg_th is not None and not math.isfinite(neg_th)):
        raise ValueError("Relation thresholds must be finite")
    if neg_th is not None and neg_th >= pos_th:
        raise ValueError("neg_th must be strictly below pos_th")
    if overlap is not None and (pos_mask is not None or neg_mask is not None):
        raise ValueError("Supply overlap or pos_mask/neg_mask, not both")
    matrix = overlap if overlap is not None else pos_mask
    if matrix is None or matrix.ndim not in (2, 3):
        raise ValueError("Relations must be [N,N] or [B,N,N]")
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("Relations must be square within each subgraph")
    batch_size, nodes_per_group = (
        (matrix.shape[0], matrix.shape[1]) if matrix.ndim == 3 else (1, len(matrix))
    )
    total = batch_size * nodes_per_group
    device = matrix.device
    valid_nodes = (
        torch.ones(total, dtype=torch.bool, device=device)
        if node_mask is None
        else node_mask.to(device).reshape(-1).bool()
    )
    if valid_nodes.numel() != total:
        raise ValueError("node_mask shape does not match embeddings")

    def relation_blocks(values, dtype):
        values = torch.as_tensor(values, device=device, dtype=dtype)
        if values.shape == (nodes_per_group, nodes_per_group) and batch_size == 1:
            return values
        if values.shape != (batch_size, nodes_per_group, nodes_per_group):
            raise ValueError(
                f"Expected relation shape {(batch_size, nodes_per_group, nodes_per_group)}, got {tuple(values.shape)}"
            )
        return torch.block_diag(*values.unbind())

    within_group = torch.block_diag(
        *[
            torch.ones(
                nodes_per_group, nodes_per_group, device=device, dtype=torch.bool
            )
            for _ in range(batch_size)
        ]
    )
    known = (
        within_group
        if pair_mask is None
        else within_group & relation_blocks(pair_mask, torch.bool)
    )
    known.fill_diagonal_(False)
    overlap_scores = (
        relation_blocks(
            overlap, matrix.dtype if matrix.is_floating_point() else torch.float32
        )
        if overlap is not None
        else torch.zeros(total, total, device=device, dtype=torch.float32)
    )
    if overlap is not None:
        if not torch.isfinite(overlap_scores[known]).all():
            raise ValueError(
                "Known overlap entries must be finite; mark unknowns with pair_mask"
            )
        positive = known & (overlap_scores >= pos_th)
        negative = known & (
            (overlap_scores < pos_th) if neg_th is None else (overlap_scores <= neg_th)
        )
    else:
        if pos_mask is None or neg_mask is None:
            raise ValueError("Supply overlap or both pos_mask and neg_mask")
        positive, negative = (
            known & relation_blocks(pos_mask, torch.bool),
            known & relation_blocks(neg_mask, torch.bool),
        )
        known = positive | negative
        if (positive & negative).any():
            raise ValueError("Positive and negative masks must be disjoint")
    group_indices = torch.arange(batch_size, device=device).repeat_interleave(
        nodes_per_group
    )[valid_nodes]
    scene_ids_flat = _identities(
        scene_ids, batch_size, nodes_per_group, valid_nodes, device, "scene_ids"
    )
    sample_ids_flat = _identities(
        sample_ids, batch_size, nodes_per_group, valid_nodes, device, "sample_ids"
    )

    def select_valid_pairs(values):
        return values[valid_nodes][:, valid_nodes]

    result = RelationSupervision(
        select_valid_pairs(positive),
        select_valid_pairs(negative),
        select_valid_pairs(known),
        group_indices,
        scene_ids_flat,
        sample_ids_flat,
        batch_size,
        torch.where(valid_nodes)[0],
        (batch_size, nodes_per_group),
        exclusive_scenes,
    )
    return exclude_duplicate_images(result)


def exclude_duplicate_images(batch: RelationSupervision) -> RelationSupervision:
    """Exclude physical-image duplicates and reject contradictory static identities."""
    positive, negative, known_relations = batch.positive, batch.negative, batch.known
    scene_ids, sample_ids = batch.scene_ids, batch.sample_ids
    same_sample = (
        (scene_ids[:, None] == scene_ids[None, :])
        & (scene_ids[:, None] >= 0)
        & (sample_ids[:, None] == sample_ids[None, :])
        & (sample_ids[:, None] >= 0)
    )
    positive, negative, known_relations = (
        positive & ~same_sample,
        negative & ~same_sample,
        known_relations & ~same_sample,
    )
    different_scenes = (
        (scene_ids[:, None] != scene_ids[None, :])
        & (scene_ids[:, None] >= 0)
        & (scene_ids[None, :] >= 0)
    )
    if (positive & negative).any() or (
        batch.exclusive_scenes and (positive & different_scenes).any()
    ):
        raise ValueError("Scene identities contradict positive GT")
    return batch._replace(positive=positive, negative=negative, known=known_relations)


def validate_relation_identities(relations: RelationSupervision) -> None:
    """Validate prepared [M] identity/group fields before candidate conversion.

    Scene/sample IDs use signed int64's range, with negatives unknown. Groups
    instead address [0,num_groups). Directly constructed supervision must honor
    the same contract as the data adapter; casting cannot repair invalid values.
    """
    if type(relations.exclusive_scenes) is not bool:
        raise TypeError("exclusive_scenes must be bool")
    group_count = integer_tensor(relations.num_groups, name="num_groups", minimum=0)
    if group_count.ndim != 0:
        raise ValueError("num_groups must be a scalar integer")
    count = len(relations.row_indices)
    for name in ("scene_ids", "sample_ids", "groups"):
        identity = getattr(relations, name)
        if not torch.is_tensor(identity) or identity.shape != (count,):
            raise ValueError(f"{name} must align with prepared descriptor rows [M]")
        bounds = (
            dict(minimum=0, maximum=int(group_count) - 1) if name == "groups" else {}
        )
        integer_tensor(identity, name=name, **bounds)


def continuous_relation_scores(
    scores: torch.Tensor, relations: RelationSupervision
) -> torch.Tensor:
    """Select observed continuous scores [M,M] using the same valid-row mapping.

    Only a custom objective that needs continuous supervision should receive
    this output. Unknown values are zeroed and cannot enter its known mask.
    """
    blocks = scores.unsqueeze(0) if scores.ndim == 2 else scores
    if (
        tuple(blocks.shape[:2]) != relations.padded_shape
        or blocks.shape[-1] != relations.padded_shape[1]
    ):
        raise ValueError("Continuous scores differ from the prepared relation layout")
    flat = torch.block_diag(*blocks.unbind())
    rows = relations.row_indices
    return flat[rows][:, rows].masked_fill(~relations.known, 0)


def prepare_relation_supervision(
    overlap: torch.Tensor,
    *,
    node_mask: torch.Tensor | None = None,
    pair_mask: torch.Tensor | None = None,
    scene_ids=None,
    sample_ids=None,
    pos_th: float = 0.3,
    neg_th: float | None = None,
    exclusive_scenes: bool = False,
    include_pair_weights: bool = False,
    positive_gamma: float = 2.0,
) -> dict[str, object]:
    """Prepare named static GT in the valid model-input row order.

    relations contains observed positive/negative masks and stable identities.
    scores/known are continuous [M,M] annotations; groups [M] addresses the
    original sampled subgraphs. Names describe data, not consuming objectives.
    Experiment bindings decide which fields a metric or regularizer receives.
    Optional pair_weights apply the explicit data-side positive-score policy.
    """
    relations = prepare_relations(
        overlap,
        node_mask=node_mask,
        pair_mask=pair_mask,
        scene_ids=scene_ids,
        sample_ids=sample_ids,
        pos_th=pos_th,
        neg_th=neg_th,
        exclusive_scenes=exclusive_scenes,
    )
    scores = continuous_relation_scores(overlap, relations)
    supervision = {
        "relations": relations,
        "scores": scores,
        "known": relations.known,
        "groups": relations.groups,
        "num_groups": relations.num_groups,
    }
    if include_pair_weights:
        supervision["pair_weights"] = positive_pair_weights(scores, positive_gamma)
    return supervision


def positive_pair_weights(scores: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Map finite continuous annotation scores [A,R] to nonnegative weights.

    This data-side policy preserves the established clamp(scores, 0)**gamma
    recipe. Targets consume these weights without interpreting overlap scores.
    """
    if (
        not torch.is_tensor(scores)
        or not scores.is_floating_point()
        or scores.ndim != 2
    ):
        raise TypeError("scores must be a floating matrix [A,R]")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not 0 < float(gamma) < float("inf"):
        raise ValueError("positive_gamma must be finite and positive")
    return scores.clamp_min(0).pow(float(gamma))
