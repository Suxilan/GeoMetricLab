"""Trainer-facing contract shared by metric objectives and their compositions."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

# PML uses (anchor, positive, negative) triplets or four positive/negative pair
# index tensors. Each index is int64 [K]; pair sides may have different lengths.
IndicesTuple = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class LossInputContract:
    """Declare PML input semantics, independently of mining or reduction math.

    None means unknown: the component retains its runtime validation. Tuple
    usage describes mathematical meaning, not whether the tuple has 3/4 sides.
    sample_weights retains PML's occurrence counts and empty-tuple fallback.
    supports_ref_emb concerns explicit references, not PML's implicit self set.
    No field declares local-row or distributed-gradient equivalence.
    """

    requires_labels: bool | None = None
    tuple_usage: (
        Literal["pairs", "triplets", "sample_weights", "unsupported", "native"] | None
    ) = None
    supports_ref_emb: bool | None = None

    def __post_init__(self) -> None:
        for name in ("requires_labels", "supports_ref_emb"):
            if (
                getattr(self, name) is not None
                and type(getattr(self, name)) is not bool
            ):
                raise TypeError(f"{name} must be bool or None")
        if self.tuple_usage not in {
            None,
            "pairs",
            "triplets",
            "sample_weights",
            "unsupported",
            "native",
        }:
            raise ValueError("Unknown PML tuple_usage")

    @property
    def supports_relations(self) -> bool:
        """Whether label-free, explicit pair/triplet supervision is declared."""
        return self.requires_labels is False and self.tuple_usage not in {
            None,
            "unsupported",
        }


def validate_loss_inputs(
    contract: LossInputContract,
    *,
    labels_present: bool,
    tuple_present: bool,
    ref_present: bool = False,
    location: str,
) -> None:
    """Reject declared incompatibilities without rewriting PML supervision."""
    if contract.requires_labels is True and not labels_present:
        raise ValueError(f"{location}: requires class IDs (labels)")
    if contract.tuple_usage == "unsupported" and tuple_present:
        raise ValueError(f"{location}: does not support indices_tuple")
    if contract.supports_ref_emb is False and ref_present:
        raise ValueError(f"{location}: does not support explicit ref_emb")


class ContextualLoss(nn.Module, ABC):
    """Accept embeddings, supervision and optional collective context.

    Implementations return a dictionary containing a scalar differentiable loss.
    Native PML objectives consume Tensor embeddings and class IDs or explicit
    pair/triplet indices. Training components may consume prepared candidates.
    A named-head objective instead consumes an embedding mapping and a
    keyword-only supervision mapping; labels never changes its class-ID meaning.
    Each objective owns its normalization, candidate and distributed behavior.
    """

    # Optional same-name inputs consumed from a shared supervision mapping.
    # Explicit output/GT remapping remains the responsibility of BoundObjective.
    supervision_keys: tuple[str, ...] | None = None

    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor | Mapping[str, torch.Tensor],
        labels: torch.Tensor | None = None,
        **context: Any,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError
