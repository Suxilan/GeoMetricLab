"""PML miners over explicit, possibly nontransitive candidate relations."""

import torch
from pytorch_metric_learning import miners

from ..losses.contracts import IndicesTuple


def validate_candidates(embeddings, references, positive, negative):
    """Validate boolean [A,R] eligible masks in the actual descriptor row spaces."""
    if embeddings.ndim != 2 or references.ndim != 2:
        raise ValueError("Candidate descriptors must be [A,D] and [R,D]")
    if (
        embeddings.shape[1] != references.shape[1]
        or embeddings.device != references.device
    ):
        raise ValueError("Anchor/reference dimensions and devices must agree")
    for mask in (positive, negative):
        if mask.dtype != torch.bool or mask.shape != (len(embeddings), len(references)):
            raise ValueError("Candidate masks must be boolean [A,R]")
        if mask.device != embeddings.device:
            raise ValueError("Candidate masks must share the descriptor device")
    if (positive & negative).any():
        raise ValueError("Positive and negative candidates must be disjoint")


def validate_mined_indices(indices, positive, negative) -> IndicesTuple:
    """Reject malformed, unknown or wrong-polarity pairs; never silently filter."""
    if not isinstance(indices, tuple) or len(indices) not in (3, 4):
        raise ValueError("Miner must return a PML pair or triplet tuple")
    anchors = (0,) if len(indices) == 3 else (0, 2)
    for axis, index in enumerate(indices):
        limit = positive.shape[0 if axis in anchors else 1]
        if (
            not torch.is_tensor(index)
            or index.ndim != 1
            or index.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Mined indices must be one-dimensional integer tensors")
        if index.device != positive.device:
            raise ValueError("Mined indices must share the candidate device")
        if (index < 0).any() or (index >= limit).any():
            raise ValueError("Mined indices exceed their anchor/reference row space")
    if len(indices) == 3:
        a, p, n = indices
        if not len(a) == len(p) == len(n):
            raise ValueError("Triplet sides must have matching lengths")
        b = a
    else:
        a, p, b, n = indices
        if len(a) != len(p) or len(b) != len(n):
            raise ValueError("Pair sides must have matching lengths")
    if not positive[a, p].all() or not negative[b, n].all():
        raise ValueError("Miner selected an unknown or wrong-polarity candidate")
    return indices


def _multi_similarity(miner, embeddings, references, positive, negative):
    values = miner.distance(embeddings, references)
    similarities = values if miner.distance.is_inverted else -values
    eligible = positive.any(1) & negative.any(1)
    hardest_negative = similarities.masked_fill(~negative, -torch.inf).amax(1)
    hardest_positive = similarities.masked_fill(~positive, torch.inf).amin(1)
    keep_positive = (
        positive
        & eligible[:, None]
        & (similarities - miner.epsilon < hardest_negative[:, None])
    )
    keep_negative = (
        negative
        & eligible[:, None]
        & (similarities + miner.epsilon > hardest_positive[:, None])
    )
    return (*torch.where(keep_positive), *torch.where(keep_negative))


def _pair_margin(miner, embeddings, references, positive, negative):
    values = miner.distance(embeddings, references)
    a, p = torch.where(positive)
    b, n = torch.where(negative)
    pos, neg = values[a, p], values[b, n]
    miner.set_stats(pos, neg)
    keep_pos = (
        pos < miner.pos_margin if miner.distance.is_inverted else pos > miner.pos_margin
    )
    keep_neg = (
        neg > miner.neg_margin if miner.distance.is_inverted else neg < miner.neg_margin
    )
    return a[keep_pos], p[keep_pos], b[keep_neg], n[keep_neg]


def _triplet_margin(miner, embeddings, references, positive, negative):
    values = miner.distance(embeddings, references)
    parts = []
    # Enumerate observed candidates per anchor, not a dense [A,R,R] temporary.
    for anchor in range(len(embeddings)):
        p = torch.where(positive[anchor])[0]
        n = torch.where(negative[anchor])[0]
        p, n = p.repeat_interleave(len(n)), n.repeat(len(p))
        parts.append((torch.full_like(p, anchor), p, n))
    a, p, n = (torch.cat([part[axis] for part in parts]) for axis in range(3))
    pos, neg = values[a, p], values[a, n]
    margin = pos - neg if miner.distance.is_inverted else neg - pos
    # Native empty-triplet statistics take mean(empty). Keep the reset zero
    # instead, so diagnostic values stay finite on empty candidate populations.
    if len(a):
        miner.set_stats(pos, neg, margin)
    if miner.type_of_triplets == "easy":
        keep = margin > miner.margin
    else:
        keep = margin <= miner.margin
        if miner.type_of_triplets == "hard":
            keep &= margin <= 0
        elif miner.type_of_triplets == "semihard":
            keep &= margin > 0
        elif miner.type_of_triplets != "all":
            raise ValueError("Unknown triplet mining strategy")
    return a[keep], p[keep], n[keep]


def _batch_easy_hard(miner, embeddings, references, positive, negative):
    """Reuse PML's selection, distance direction, ranges and statistics methods."""
    values = miner.distance(embeddings, references)
    positive_anchors, positive_references = torch.where(positive)
    negative_anchors, negative_references = torch.where(negative)
    rows = torch.arange(len(embeddings), device=embeddings.device)
    if miner.pos_strategy == miner.SEMIHARD:
        (negative_distances, negative_indices), negative_valid = miner.get_negatives(
            values, negative_anchors, negative_references
        )
        (positive_distances, positive_indices), positive_valid = miner.get_positives(
            values, positive_anchors, positive_references, negative_distances
        )
    elif miner.neg_strategy == miner.SEMIHARD:
        (positive_distances, positive_indices), positive_valid = miner.get_positives(
            values, positive_anchors, positive_references
        )
        (negative_distances, negative_indices), negative_valid = miner.get_negatives(
            values, negative_anchors, negative_references, positive_distances
        )
    else:
        if miner.pos_strategy != miner.ALL:
            (positive_distances, positive_indices), positive_valid = (
                miner.get_positives(values, positive_anchors, positive_references)
            )
        if miner.neg_strategy != miner.ALL:
            (negative_distances, negative_indices), negative_valid = (
                miner.get_negatives(values, negative_anchors, negative_references)
            )
    if miner.ALL not in (miner.pos_strategy, miner.neg_strategy):
        valid = positive_valid & negative_valid
        miner.set_stats(positive_distances[valid], negative_distances[valid])
        positive_anchors, positive_references = rows[valid], positive_indices[valid]
        negative_anchors, negative_references = rows[valid], negative_indices[valid]
    elif miner.pos_strategy == miner.ALL and miner.neg_strategy != miner.ALL:
        miner.set_stats(
            values[positive_anchors, positive_references],
            negative_distances[negative_valid],
        )
        negative_anchors = rows[negative_valid]
        negative_references = negative_indices[negative_valid]
    elif miner.pos_strategy != miner.ALL and miner.neg_strategy == miner.ALL:
        miner.set_stats(
            positive_distances[positive_valid],
            values[negative_anchors, negative_references],
        )
        positive_anchors = rows[positive_valid]
        positive_references = positive_indices[positive_valid]
    else:
        miner.set_stats(
            values[positive_anchors, positive_references],
            values[negative_anchors, negative_references],
        )
    if type(miner) is miners.BatchHardMiner:
        return positive_anchors, positive_references, negative_references
    return positive_anchors, positive_references, negative_anchors, negative_references


# Native PML has no mask argument. These fixed adapters supply only that missing
# candidate boundary. Custom components implement mine_candidates directly;
# they never enter a second name/factory registry or a parameter-field table.
_NATIVE_ADAPTERS = {
    miners.MultiSimilarityMiner: _multi_similarity,
    miners.PairMarginMiner: _pair_margin,
    miners.TripletMarginMiner: _triplet_margin,
    miners.BatchEasyHardMiner: _batch_easy_hard,
    miners.BatchHardMiner: _batch_easy_hard,
}


def relation_miner_adapter(miner):
    """Validate relation support and return a native candidate adapter, if any.

    A custom candidate hook takes precedence. Labels-only mining must not be
    invoked with fabricated classes for a nontransitive relation graph.
    """
    adapter = _NATIVE_ADAPTERS.get(type(miner))
    if adapter is None and (
        isinstance(miner, miners.MultiSimilarityMiner)
        and type(miner).mine is miners.MultiSimilarityMiner.mine
        and type(miner).forward is miners.MultiSimilarityMiner.forward
    ):
        adapter = _multi_similarity
    if not callable(getattr(miner, "mine_candidates", None)) and adapter is None:
        raise ValueError(
            f"{type(miner).__name__} has no relation-candidate implementation; "
            "provide class-label supervision or a mine_candidates extension"
        )
    return adapter


def mine_candidates(
    miner: miners.BaseMiner,
    embeddings: torch.Tensor,
    positive: torch.Tensor | None = None,
    negative: torch.Tensor | None = None,
    *,
    labels: torch.Tensor | None = None,
    indices_tuple: IndicesTuple | None = None,
    ref_emb: torch.Tensor | None = None,
) -> IndicesTuple:
    """Mine labels natively, or select within explicit candidate masks/tuples.

    Unrestricted labels use BaseMiner.forward. Explicit tuples currently support
    native MS (including subclasses inheriting its computation unchanged). MS
    filters original occurrences; exact triplets survive only when both sides
    pass selection, without generating additional triplet combinations. Custom
    candidate hooks are supported only by the mask entry, not by explicit tuples.

    Custom BaseMiner subclasses may implement
    mine_candidates(embeddings, positive, negative, ref_emb=...). Standard mine
    remains the label extension point. This bridge supplies the reset/no-grad/
    output checks that BaseMiner.forward cannot perform without class labels.
    Unknown relations are absent from both masks; no class IDs are fabricated.
    """
    if indices_tuple is None and positive is None and negative is None:
        # Keep PML's native lifecycle and arbitrary BaseMiner extensions intact.
        return miner(embeddings, labels, ref_emb=ref_emb)
    if indices_tuple is not None:
        if not isinstance(miner, miners.BaseMiner):
            raise TypeError("Candidate mining requires a PML BaseMiner")
        miner.reset_stats()
        if positive is not None or negative is not None:
            raise ValueError("Specify candidate masks or indices_tuple, not both")
        with torch.no_grad():
            indices = _mine_tuple(miner, embeddings, indices_tuple, ref_emb)
        miner.output_assertion(indices)
        return indices
    if not isinstance(miner, miners.BaseMiner):
        raise TypeError("Candidate mining requires a PML BaseMiner")
    if positive is None or negative is None:
        raise ValueError("Provide both positive and negative candidate masks")
    references = embeddings if ref_emb is None else ref_emb
    validate_candidates(embeddings, references, positive, negative)
    custom = getattr(miner, "mine_candidates", None)
    adapter = relation_miner_adapter(miner)
    miner.reset_stats()
    with torch.no_grad():
        if callable(custom):
            indices = custom(
                embeddings, positive.clone(), negative.clone(), ref_emb=references
            )
        elif positive.numel() == 0:
            count = (
                3
                if type(miner) in (miners.TripletMarginMiner, miners.BatchHardMiner)
                else 4
            )
            indices = tuple(
                torch.empty(0, device=embeddings.device, dtype=torch.long)
                for _ in range(count)
            )
        else:
            indices = adapter(miner, embeddings, references, positive, negative)
        indices = validate_mined_indices(indices, positive, negative)
    miner.output_assertion(indices)
    return indices


def _mine_tuple(miner, embeddings, candidates, ref_emb):
    """Filter candidate occurrences without recombining exact triplets.

    MS uses a binary projection for selection only. Returned indices retain
    input order, direction and multiplicity. Algorithms with occurrence quotas
    require a separate, verified candidate adapter rather than deduplication.
    """
    references = embeddings if ref_emb is None else ref_emb
    if not isinstance(candidates, (tuple, list)) or len(candidates) not in (3, 4):
        raise ValueError("Candidates must be a PML pair or triplet tuple")
    shape = (len(embeddings), len(references))
    positive = torch.zeros(shape, dtype=torch.bool, device=embeddings.device)
    negative = torch.zeros_like(positive)
    for axis, index in enumerate(candidates):
        anchor_axis = axis == 0 or (len(candidates) == 4 and axis == 2)
        limit = shape[0 if anchor_axis else 1]
        if (
            not torch.is_tensor(index)
            or index.ndim != 1
            or index.dtype not in (torch.int32, torch.int64)
            or index.device != embeddings.device
        ):
            raise ValueError(
                "Candidates must be one-dimensional integer tensors on the descriptor device"
            )
        if (index < 0).any() or (index >= limit).any():
            raise ValueError("Candidates exceed their anchor/reference row space")
    if len(candidates) == 3:
        anchors, positives, negatives = candidates
        if not len(anchors) == len(positives) == len(negatives):
            raise ValueError("Triplet sides must have matching lengths")
        negative_anchors = anchors
    else:
        anchors, positives, negative_anchors, negatives = candidates
        if len(anchors) != len(positives) or len(negative_anchors) != len(negatives):
            raise ValueError("Pair sides must have matching lengths")
    positive[anchors, positives] = True
    negative[negative_anchors, negatives] = True
    native_ms = (
        isinstance(miner, miners.MultiSimilarityMiner)
        and type(miner).mine is miners.MultiSimilarityMiner.mine
        and type(miner).forward is miners.MultiSimilarityMiner.forward
    )
    if not native_ms:
        raise ValueError(
            "Explicit candidate tuples currently require MultiSimilarityMiner"
        )
    if callable(getattr(miner, "mine_candidates", None)):
        raise ValueError(
            "Custom mine_candidates hooks do not support explicit candidate tuples"
        )
    validate_candidates(embeddings, references, positive, negative)
    if positive.numel() == 0:
        return tuple(index[:0] for index in candidates)
    selected = _multi_similarity(miner, embeddings, references, positive, negative)
    positive.zero_()
    negative.zero_()
    positive[selected[0], selected[1]] = True
    negative[selected[2], selected[3]] = True
    keep_positive = positive[anchors, positives]
    keep_negative = negative[negative_anchors, negatives]
    if len(candidates) == 3:
        keep = keep_positive & keep_negative
        return tuple(index[keep] for index in candidates)
    return (
        anchors[keep_positive],
        positives[keep_positive],
        negative_anchors[keep_negative],
        negatives[keep_negative],
    )
