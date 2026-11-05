"""Native PML evaluation over labels and explicit candidate indices."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .contracts import ContextualLoss, validate_loss_inputs
from .registry import build_loss


def pair_weight_arguments(loss, embeddings, ref_emb, pair_weights):
    """Validate consumed, finite nonnegative [anchors,references] weights."""
    if pair_weights is None:
        return {}
    if not getattr(loss, "supports_pair_weights", False):
        raise ValueError("This objective does not support pair_weights")
    if not torch.is_tensor(pair_weights) or not pair_weights.is_floating_point():
        raise TypeError("pair_weights must be a floating Tensor [A,R]")
    references = embeddings if ref_emb is None else ref_emb
    if pair_weights.shape != (len(embeddings), len(references)):
        raise ValueError("pair_weights must align with anchors/references [A,R]")
    if not torch.isfinite(pair_weights).all() or (pair_weights < 0).any():
        raise ValueError("pair_weights must be finite and nonnegative")
    return {"pair_weights": pair_weights}


def evaluate_pml_loss(
    loss,
    embeddings,
    labels=None,
    *,
    indices_tuple=None,
    ref_emb=None,
    pair_weights=None,
):
    """Call the configured loss without replacing its math, forward, or reducer."""
    arguments = pair_weight_arguments(loss, embeddings, ref_emb, pair_weights)
    return loss(
        embeddings, labels, indices_tuple=indices_tuple, ref_emb=ref_emb, **arguments
    )


class PMLObjective(ContextualLoss):
    """Evaluate a native PML loss on already selected labels/pairs/triplets.

    Mining and distributed candidate preparation belong to the calling training
    component. normalization is optional and independent of PML's distance.
    loss_fn_name accepts a native/registered name, type mapping or module instance.
    Only names accept loss_kwargs. Native loss mathematics and reducers are retained.
    """

    def __init__(
        self,
        loss_fn_name: str | nn.Module | Mapping[str, Any] = "MultiSimilarityLoss",
        loss_kwargs: Mapping[str, Any] | None = None,
        *,
        normalize_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.base_loss, self.loss_fn_name, self.input_contract = build_loss(
            loss_fn_name, dict(loss_kwargs or {}), None, None
        )
        self.supports_relations = self.input_contract.supports_relations
        if type(normalize_embeddings) is not bool:
            raise TypeError("normalize_embeddings must be bool")
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        embeddings,
        labels=None,
        *,
        indices_tuple=None,
        ref_emb=None,
        pair_weights=None,
    ):
        """Return scalar loss for float [A,D], IDs [A] or explicit PML indices.

        pair_weights is finite nonnegative [A,R] supervision for objectives that
        declare support, in both ID and tuple modes. Unsupported inputs fail.
        """
        if embeddings.ndim != 2 or not embeddings.is_floating_point():
            raise ValueError("embeddings must be floating [N,D]")
        validate_loss_inputs(
            self.input_contract,
            labels_present=labels is not None,
            tuple_present=indices_tuple is not None,
            ref_present=ref_emb is not None,
            location=f"PMLObjective.loss_fn_name={self.loss_fn_name}",
        )
        if labels is not None and (
            labels.ndim != 1
            or len(labels) != len(embeddings)
            or labels.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "Class labels must be integer [N], aligned with embeddings"
            )
        with torch.autocast(embeddings.device.type, enabled=False):
            values = (
                embeddings.float()
                if embeddings.dtype in (torch.float16, torch.bfloat16)
                else embeddings
            )
            references = ref_emb
            if references is not None and references.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                references = references.float()
            if self.normalize_embeddings:
                values = torch.nn.functional.normalize(values, dim=-1, eps=1e-6)
                if references is not None:
                    references = torch.nn.functional.normalize(
                        references, dim=-1, eps=1e-6
                    )
            loss = evaluate_pml_loss(
                self.base_loss,
                values,
                labels,
                indices_tuple=indices_tuple,
                ref_emb=references,
                pair_weights=pair_weights,
            )
        return {"loss": loss}
