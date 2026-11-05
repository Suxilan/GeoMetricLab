"""Overlap ordering expressed as PML triplets, with bounded temporary memory."""

import math
from numbers import Integral

import torch
from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import MeanReducer
from torch import nn
from torch.nn import functional as F

from ..contracts import ContextualLoss


def _validate_ranking_inputs(embeddings, scores, known, groups, num_groups):
    """Reject misaligned supervision rather than silently dropping group rows."""
    count = len(embeddings)
    if (
        embeddings.ndim != 2
        or not embeddings.is_floating_point()
        or scores.shape != (count, count)
        or known.shape != scores.shape
        or groups.shape != (count,)
    ):
        raise ValueError("Ranking supervision must align with floating descriptors")
    if any(value.device != embeddings.device for value in (scores, known, groups)):
        raise ValueError("Ranking inputs must share the descriptor device")
    if known.dtype != torch.bool or not torch.isfinite(scores[known]).all():
        raise ValueError("Ranking requires boolean known pairs and finite scores")
    if (
        isinstance(num_groups, bool)
        or not isinstance(num_groups, Integral)
        or num_groups < 0
    ):
        raise ValueError("num_groups must be a nonnegative integer")
    if groups.dtype not in (torch.int32, torch.int64):
        raise ValueError("Ranking groups must be integer [M]")
    if (groups < 0).any() or (groups >= num_groups).any():
        raise ValueError("Ranking groups must lie in [0,num_groups)")


class OverlapRankingLoss(nn.Module):
    """Rank unit descriptors by overlap within each observed subgraph.

    A triplet (a,p,n) is eligible when overlap[a,p]-overlap[a,n] > ov_margin.
    Its hinge is max(0, sim_margin-sim[a,p]+sim[a,n]); the mean includes zero
    hinges. Chunking bounds temporary triplet masks without changing the target.
    """

    def __init__(
        self,
        ov_margin: float = 0.05,
        sim_margin: float = 0.05,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, Integral)
            or chunk_size < 1
        ):
            raise ValueError("rank chunk_size must be a positive integer")
        self.ov_margin = float(ov_margin)
        self.sim_margin = float(sim_margin)
        if not math.isfinite(self.ov_margin) or not math.isfinite(self.sim_margin):
            raise ValueError("Ranking margins must be finite")
        self.chunk_size = int(chunk_size)
        # The caller supplies unit descriptors; keep dot-product gradients intact.
        self.loss = TripletMarginLoss(
            margin=sim_margin,
            distance=DotProductSimilarity(normalize_embeddings=False),
            reducer=MeanReducer(),
        )

    def statistics(
        self,
        embeddings: torch.Tensor,
        overlap: torch.Tensor,
        known: torch.Tensor,
        groups: torch.Tensor,
        num_groups: int,
    ) -> tuple[torch.Tensor, int]:
        """Return the hinge sum and eligible triplet count across all subgraphs."""
        _validate_ranking_inputs(embeddings, overlap, known, groups, num_groups)
        total = embeddings.sum() * 0
        total_count = 0
        for group in range(num_groups):
            group_indices = torch.where(groups == group)[0]
            descriptors = embeddings[group_indices]
            overlap_scores = overlap[group_indices[:, None], group_indices[None, :]]
            known_pairs = known[group_indices[:, None], group_indices[None, :]]
            group_numerator = embeddings.sum() * 0
            group_count = 0
            for anchor_start in range(0, len(group_indices), self.chunk_size):
                anchor_stop = min(anchor_start + self.chunk_size, len(group_indices))
                anchor_pairs = known_pairs[anchor_start:anchor_stop]
                anchor_scores = overlap_scores[anchor_start:anchor_stop]
                ordering_mask = (
                    anchor_pairs[:, :, None]
                    & anchor_pairs[:, None, :]
                    & (
                        (anchor_scores[:, :, None] - anchor_scores[:, None, :])
                        > self.ov_margin
                    )
                )
                anchors, positive_references, negative_references = torch.where(
                    ordering_mask
                )
                triplet_count = anchors.numel()
                if triplet_count:
                    ranking_loss = self.loss(
                        descriptors[anchor_start:anchor_stop],
                        indices_tuple=(
                            anchors,
                            positive_references,
                            negative_references,
                        ),
                        ref_emb=descriptors,
                    )
                    group_numerator = group_numerator + ranking_loss * triplet_count
                    group_count += triplet_count
            total = total + group_numerator
            total_count += group_count
        return total, total_count

    def forward(
        self,
        embeddings: torch.Tensor,
        overlap: torch.Tensor,
        known: torch.Tensor,
        groups: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """Return the mean hinge over eligible within-group triplets.

        Descriptors are unit [M,D]; overlap and observed-pair mask are [M,M].
        Integer groups [M] maps every row into [0,num_groups). Empty targets return a
        differentiable zero; unknown relations do not enter the triplet count.
        """
        total, triplet_count = self.statistics(
            embeddings, overlap, known, groups, num_groups
        )
        return total / max(triplet_count, 1)


class OverlapRankingObjective(ContextualLoss):
    """Adapt overlap ranking to the trainer contract, independently of metrics.

    Ordering and triplet averaging are local to each rank, within sampled graphs.
    Lightning owns DDP parameter-gradient averaging; this loss has no collectives.
    """

    supervision_keys = ("scores", "known", "groups", "num_groups")

    def __init__(
        self,
        normalize_embeddings: bool = True,
        *,
        ov_margin: float = 0.05,
        sim_margin: float = 0.05,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.ranking = OverlapRankingLoss(
            ov_margin=ov_margin,
            sim_margin=sim_margin,
            chunk_size=chunk_size,
        )
        if type(normalize_embeddings) is not bool:
            raise TypeError("normalize_embeddings must be bool")
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        embeddings,
        labels=None,
        *,
        scores,
        known,
        groups,
        num_groups,
    ):
        """Rank valid descriptors [M,D] using aligned continuous supervision.

        scores/known are [M,M]; groups [M] selects sampled subgraphs. The caller
        has already removed padding, self and duplicate-image comparisons.
        This objective interprets ordering margins, never dataset GT thresholds.
        """
        if labels is not None:
            raise ValueError("Ranking requires explicit continuous supervision")
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            values = (
                embeddings.float()
                if embeddings.dtype in (torch.float16, torch.bfloat16)
                else embeddings
            )
            if self.normalize_embeddings:
                values = F.normalize(values, dim=-1, eps=1e-6)
            numerator, triplet_count = self.ranking.statistics(
                values,
                scores.to(values),
                known.to(values.device),
                groups.to(values.device),
                num_groups,
            )
            loss = numerator / max(triplet_count, 1)
        return {"loss": loss}
