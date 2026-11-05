"""Descriptor candidates and index remapping for prepared metric supervision."""

from typing import NamedTuple

import torch

from ..datasets.utils.relations import RelationSupervision, validate_relation_identities


class RelationBatch(NamedTuple):
    """Flatten valid nodes from [B,N,D] into aligned [M,D] descriptors.

    positive/negative/known/pair_weights are [M,M]; groups maps each node to its
    input subgraph. scene_ids/sample_ids identify physical images; negative
    values denote unknown identity. Different subgraphs of one scene do not imply
    observed cross-subgraph relations.
    """

    embeddings: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    pair_weights: torch.Tensor
    known: torch.Tensor
    groups: torch.Tensor
    scene_ids: torch.Tensor
    sample_ids: torch.Tensor
    num_groups: int


class RelationReferences(NamedTuple):
    """Candidate descriptors [R,D] with aligned scene/sample identities [R].

    local_offset locates the local node block inside the reference collection.
    Remote relations require an explicit cross-scene policy; identity alone
    never supplies missing positive GT.
    """

    embeddings: torch.Tensor
    scene_ids: torch.Tensor
    sample_ids: torch.Tensor
    local_offset: int


def local_references(batch: RelationBatch) -> RelationReferences:
    """Use the batch's [M,D] descriptors as its reference set at offset zero."""
    return RelationReferences(batch.embeddings, batch.scene_ids, batch.sample_ids, 0)


def anchor_relations(
    batch: RelationBatch,
    references: RelationReferences,
    cross_scene_negatives: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive/negative masks [A,R] for local anchors and references.

    Insert only the observed local GT block. Cross-scene negatives require known
    mutually exclusive scene IDs; equal physical-image identities are excluded.
    Missing relations within one scene remain unknown rather than negative.
    """
    local_count = len(batch.embeddings)
    reference_count = len(references.embeddings)
    positive = torch.zeros(
        local_count,
        reference_count,
        device=batch.embeddings.device,
        dtype=torch.bool,
    )
    negative = torch.zeros_like(positive)
    offset = references.local_offset
    positive[:, offset : offset + local_count] = batch.positive
    negative[:, offset : offset + local_count] = batch.negative
    scenes = batch.scene_ids[:, None]
    ref_scenes = references.scene_ids[None, :]
    if cross_scene_negatives:
        negative |= (scenes != ref_scenes) & (scenes >= 0) & (ref_scenes >= 0)
    same_image = (
        (scenes == ref_scenes)
        & (scenes >= 0)
        & (batch.sample_ids[:, None] == references.sample_ids[None, :])
        & (batch.sample_ids[:, None] >= 0)
        & (references.sample_ids[None, :] >= 0)
    )
    positive &= ~same_image
    negative &= ~same_image
    if (positive & negative).any():
        raise ValueError("Scene identities contradict positive GT")
    return positive, negative


def bind_relations(
    embeddings: torch.Tensor, relations: RelationSupervision, pair_weights=None
) -> RelationBatch:
    """Bind already unpadded [M,D] descriptors to data-prepared supervision."""
    count = len(relations.row_indices)
    if embeddings.ndim != 2 or len(embeddings) != count:
        raise ValueError("Descriptors must be [M,D] in prepared valid-row order")
    validate_relation_identities(relations)
    if pair_weights is None:
        pair_weights = embeddings.new_zeros((count, count))
    elif tuple(pair_weights.shape) != (count, count):
        raise ValueError("Pair weights must align with valid descriptors [M,M]")
    return RelationBatch(
        embeddings,
        relations.positive.to(embeddings.device),
        relations.negative.to(embeddings.device),
        pair_weights.to(embeddings),
        relations.known.to(embeddings.device),
        relations.groups.to(device=embeddings.device, dtype=torch.int64),
        relations.scene_ids.to(device=embeddings.device, dtype=torch.int64),
        relations.sample_ids.to(device=embeddings.device, dtype=torch.int64),
        int(relations.num_groups),
    )


def apply_candidate_policy(
    batch: RelationBatch, exclusive_scenes: bool
) -> RelationBatch:
    """Add only the cross-scene negatives declared by the data contract."""
    positive, negative = anchor_relations(
        batch, local_references(batch), exclusive_scenes
    )
    return batch._replace(positive=positive, negative=negative)
