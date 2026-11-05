"""PML tuple conversion in explicit anchor/reference row spaces."""

import torch

from ..losses.contracts import IndicesTuple


def pair_indices(
    positive_mask: torch.Tensor, negative_mask: torch.Tensor
) -> IndicesTuple:
    """Convert positive/negative masks [A,R] to PML's four pair-index tensors."""
    return (*torch.where(positive_mask), *torch.where(negative_mask))


def pairs_to_masks(
    indices: IndicesTuple, shape: tuple[int, int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode PML triplets (a,p,n) or pairs (a,p,b,n) as boolean [A,R] masks."""
    positive_mask = torch.zeros(shape, device=device, dtype=torch.bool)
    negative_mask = torch.zeros_like(positive_mask)
    if len(indices) == 3:
        positive_anchors, positive_references, negative_references = indices
        positive_mask[positive_anchors, positive_references] = True
        negative_mask[positive_anchors, negative_references] = True
    elif len(indices) == 4:
        positive_anchors, positive_references, negative_anchors, negative_references = (
            indices
        )
        positive_mask[positive_anchors, positive_references] = True
        negative_mask[negative_anchors, negative_references] = True
    else:
        raise ValueError("PML indices must contain three triplet or four pair tensors")
    return positive_mask, negative_mask
