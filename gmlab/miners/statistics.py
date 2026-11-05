"""Detached candidate-retention reports, independent of loss mathematics."""

import torch


def pair_counts(indices):
    """Count distinct directed pairs, including pairs projected from triplets."""
    if len(indices) == 3:
        anchors, positive, negative = indices
        pairs = ((anchors, positive), (anchors, negative))
    else:
        pairs = ((indices[0], indices[1]), (indices[2], indices[3]))
    return tuple(torch.stack(pair, dim=1).unique(dim=0).shape[0] for pair in pairs)


def retention_reports(candidate_counts, selected, reference):
    """Report unique directed pair retention, never a triplet occurrence ratio.

    Empty candidate sets report zero retention. Counts describe the candidate
    domain used by the miner (global when candidates were gathered).
    """
    selected_counts = pair_counts(selected)
    total = sum(candidate_counts)
    kept = sum(selected_counts)
    values = {
        "miner/candidate_positive_pairs": candidate_counts[0],
        "miner/candidate_negative_pairs": candidate_counts[1],
        "miner/selected_positive_pairs": selected_counts[0],
        "miner/selected_negative_pairs": selected_counts[1],
        "miner/rate": kept / total if total else 0.0,
    }
    return {
        name: reference.new_tensor(value).detach() for name, value in values.items()
    }


def label_pair_counts(labels):
    """Count native non-self pairs without constructing a quadratic mask."""
    counts = labels.unique(return_counts=True)[1]
    positive = int((counts * (counts - 1)).sum())
    return positive, len(labels) * (len(labels) - 1) - positive
