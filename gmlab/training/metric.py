"""Candidate preparation and a shared PML execution path for metric learning."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
import logging

from torch import nn

import torch
from torch.nn import functional as F
from pytorch_metric_learning import miners
from pytorch_metric_learning.utils import distributed as pml_distributed

from ..datasets.utils.relations import RelationSupervision, validate_relation_identities
from ..utils.distributed import GatherFunction, check_rank_consistency
from ..losses.contracts import ContextualLoss, IndicesTuple, validate_loss_inputs
from ..losses.objectives import (
    evaluate_pml_loss,
    pair_weight_arguments,
)
from ..losses.registry import build_loss, build_pml_component
from .collectives import CandidateCollectives
from .candidates import (
    anchor_relations,
    apply_candidate_policy,
    bind_relations,
    local_references,
)
from .indices import pair_indices
from ..miners import mine_candidates
from ..miners.statistics import pair_counts, label_pair_counts, retention_reports
from ..miners.candidates import relation_miner_adapter


@dataclass
class PreparedMetric:
    """One step's PML inputs, after identity alignment and optional mining.

    embeddings [A,D] and references [R,D] define the tuple index spaces.
    labels [A] are retained for class-dependent targets. native_wrapper owns
    gathering when selected.
    """

    embeddings: torch.Tensor
    labels: torch.Tensor | None = None
    indices: IndicesTuple | None = None
    references: torch.Tensor | None = None
    pair_weights: torch.Tensor | None = None
    native_wrapper: bool = False
    mining_reports: dict | None = None


class MetricObjective(ContextualLoss):
    """Bind candidates, select tuples, and execute one configured PML target.

    Static relation semantics belong to the data adapter. Explicit tuples are
    rank-local observed pairs/triplets, never invented cross-rank relations.
    labels and tuples may coexist when supported by the injected PML target.
    Mathematical policies apply equally to every supervision representation.

    loss_fn_name/miner_name accept native or registered names, explicit type
    mappings, or constructed modules. kwargs belong only to name construction;
    a mapping carries its own arguments and an instance cannot accept overrides.
    Native loss mathematics and reducers are retained for every GT representation.

    Native candidate adapters use effective instance settings. Custom PML miners
    implement mine_candidates to accept nontransitive masks; returned indices
    are checked against those masks before any objective consumes them.

    distributed_backend constrains all representations when ranks are active:
    auto selects a compatible gather owner, autograd prepares candidates here,
    and pml requires native-wrapper eligibility. One-process execution is local.
    """

    supervision_keys = ("relations",)

    def __init__(
        self,
        loss_fn_name: str | nn.Module | Mapping[str, Any] = "MultiSimilarityLoss",
        miner_name: str | miners.BaseMiner | Mapping[str, Any] | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
        miner_kwargs: Mapping[str, Any] | None = None,
        *,
        distributed: bool = True,
        labels_scope: Literal["global", "rank"] = "global",
        normalize_embeddings: bool = True,
        custom_loss_module: str | None = None,
        custom_supports_relations: bool | None = None,
        distributed_backend: Literal["auto", "pml", "autograd"] = "auto",
    ) -> None:
        super().__init__()
        if labels_scope not in {"global", "rank"}:
            raise ValueError("labels_scope must be global or rank")
        if distributed_backend not in {"auto", "pml", "autograd"}:
            raise ValueError("distributed_backend must be auto, pml or autograd")
        self.base_loss, self.loss_fn_name, self.input_contract = build_loss(
            loss_fn_name,
            dict(loss_kwargs or {}),
            custom_loss_module,
            custom_supports_relations,
        )
        self.relation_resolvers = {}
        self.supports_relations = self.input_contract.supports_relations
        if (miner_name is None or miner_name == "no_miner") and miner_kwargs:
            raise ValueError("miner_kwargs requires an enabled miner")
        self.miner_name = None
        self.miner = None
        if miner_name is not None and miner_name != "no_miner":
            self.miner = build_pml_component(miner_name, miners, miner_kwargs)
            if not isinstance(self.miner, miners.BaseMiner):
                raise TypeError(
                    f"{self.loss_fn_name}.miner_name must implement PML BaseMiner"
                )
            self.miner_name = type(self.miner).__name__
            validate_loss_inputs(
                self.input_contract,
                labels_present=True,
                tuple_present=True,
                location=f"MetricObjective.loss_fn_name={self.loss_fn_name}.miner_name",
            )
            if self.input_contract.tuple_usage == "sample_weights":
                logging.getLogger(__name__).info(
                    "%s consumes miner tuples as PML occurrence-frequency sample "
                    "weights; an empty tuple uses all-one weights",
                    self.loss_fn_name,
                )
        if type(distributed) is not bool or type(normalize_embeddings) is not bool:
            raise TypeError("distributed and normalize_embeddings must be bool")
        self.distributed = distributed
        self.labels_scope = labels_scope
        self.distributed_backend = distributed_backend
        self.normalize_embeddings = normalize_embeddings
        self.last_backend = "local"
        self.last_pair_shape = None

    def extra_repr(self) -> str:
        return f"loss={self.loss_fn_name}, input_contract={self.input_contract}"

    def _native_wrapper_eligible(
        self, values, labels, indices, pair_weights, collectives
    ):
        """Use PML's gather owner only when it preserves the selected math."""
        if not collectives.active or self.distributed_backend == "autograd":
            return False
        if labels is None or indices is not None or pair_weights is not None:
            return False
        sizes = collectives.sizes(len(values), values.device)
        parameterless = not any(p.requires_grad for p in self.base_loss.parameters())
        return len(set(sizes)) == 1 and self.labels_scope == "global" and parameterless

    def _prepare_native(self, values, labels, indices, pair_weights, collectives):
        """Prepare labels/tuples without changing their meaning or target math."""
        if labels is not None:
            labels = torch.as_tensor(labels, device=values.device)
        elif not self.supports_relations:
            raise ValueError(f"{self.loss_fn_name} requires class IDs")
        if self._native_wrapper_eligible(
            values, labels, indices, pair_weights, collectives
        ):
            if self.miner is not None:
                indices = pml_distributed.DistributedMinerWrapper(
                    self.miner, efficient=False
                )(values, labels)
            reports = None
            if self.miner is not None:
                sizes = collectives.sizes(len(labels), labels.device)
                global_labels = torch.cat(collectives.gather_variable(labels, sizes))
                reports = retention_reports(
                    label_pair_counts(global_labels), indices, values
                )
            return PreparedMetric(
                values, labels, indices, native_wrapper=True, mining_reports=reports
            )
        if self.distributed_backend == "pml" and collectives.active:
            raise ValueError(
                "PML wrappers require native math, equal local counts, global IDs, "
                "no explicit tuples or pair_weights, and a parameterless loss; "
                "use auto or autograd"
            )
        if indices is not None:
            indices = validate_local_indices(indices, len(values), values.device)
        if collectives.active:
            sizes = collectives.sizes(len(values), values.device)
            if labels is not None:
                values, labels = collectives.ids(values, labels, self.labels_scope)
            else:
                values = torch.cat(
                    collectives.gather_variable(values, sizes, sync_grads=True)
                )
            if indices is not None:
                offset = sum(sizes[: collectives.rank])
                lengths = collectives.sizes(len(indices[0]), values.device)
                negative_lengths = (
                    lengths
                    if len(indices) == 3
                    else collectives.sizes(len(indices[2]), values.device)
                )
                indices = tuple(
                    torch.cat(
                        collectives.gather_variable(
                            index + offset,
                            lengths
                            if axis < 2 or len(indices) == 3
                            else negative_lengths,
                        )
                    )
                    for axis, index in enumerate(indices)
                )
            if pair_weights is not None:
                # Explicit tuples address only observed rank-local blocks. No
                # selected pair uses the unprovided cross-rank weight entries.
                pair_weights = torch.block_diag(
                    *collectives.gather_variable(
                        pair_weights.to(values), sizes, square=True
                    )
                )
        reports = None
        if self.miner is not None:
            candidates = (
                pair_counts(indices)
                if indices is not None
                else label_pair_counts(labels)
            )
            indices = mine_candidates(
                self.miner, values, labels=labels, indices_tuple=indices
            )
            reports = retention_reports(candidates, indices, values)
        return PreparedMetric(
            values, labels, indices, pair_weights=pair_weights, mining_reports=reports
        )

    def _mine_relations(self, embeddings, references, positive, negative):
        """Select observed tuples with the constructed PML candidate miner."""
        if self.miner is None:
            return pair_indices(positive, negative)
        return mine_candidates(
            self.miner, embeddings, positive, negative, ref_emb=references
        )

    def bind_relation_resolver(self, resolver):
        """Register a static GT query by source and decoding thresholds.

        Source keys travel with supervision, so named objectives cannot silently
        replace another GT's observed masks using a Dataset-wide default query.
        """
        if resolver.key is None:
            raise ValueError("A bound relation resolver requires a source key")
        if resolver.key in self.relation_resolvers:
            raise ValueError(f"Duplicate relation source: {resolver.key}")
        self.relation_resolvers[resolver.key] = resolver

    def _resolve_relation_gt(self, batch, source, collectives):
        if source is None:
            return batch
        query_error = None
        try:
            resolver = self.relation_resolvers[source]
            positive, negative = resolver(batch.scene_ids, batch.sample_ids)
            if (batch.positive & ~positive).any() or (batch.negative & ~negative).any():
                raise ValueError("Full GT query contradicts observed batch supervision")
        except (ValueError, KeyError, IndexError) as error:
            query_error = (
                f"ValueError: Cannot resolve relation source {source!r}: {error}"
            )
        check_rank_consistency(query_error, None, enabled=collectives.active)
        if query_error is not None:
            raise ValueError(query_error)
        return batch._replace(
            positive=positive, negative=negative, known=positive | negative
        )

    def _prepare_relations(self, values, relations, pair_weights, collectives):
        """Bind static masks to local/global reference rows, then select candidates."""
        if not self.supports_relations:
            raise ValueError(f"{self.loss_fn_name} requires class IDs")
        batch = bind_relations(values, relations, pair_weights)
        if collectives.active:
            batch = collectives.global_relations(batch)
        batch = self._resolve_relation_gt(batch, relations.source, collectives)
        batch = apply_candidate_policy(batch, relations.exclusive_scenes)
        references = local_references(batch)
        positive, negative = anchor_relations(
            batch, references, relations.exclusive_scenes
        )
        self.last_pair_shape = tuple(positive.shape)
        mining_error = None
        indices = None
        try:
            indices = self._mine_relations(
                batch.embeddings, references.embeddings, positive, negative
            )
        except (ValueError, TypeError, IndexError, AssertionError) as exc:
            mining_error = f"{type(exc).__name__}: {exc}"
        # A rejected local extension must not strand another rank in the next
        # differentiable reduction. Reuse the existing rank error boundary.
        check_rank_consistency(
            mining_error, None, enabled=collectives.active and self.miner is not None
        )
        weights = None
        if pair_weights is not None:
            weights = batch.embeddings.new_zeros(positive.shape)
            offset = references.local_offset
            weights[:, offset : offset + len(batch.embeddings)] = batch.pair_weights
        return PreparedMetric(
            batch.embeddings,
            indices=indices,
            references=references.embeddings,
            pair_weights=weights,
            mining_reports=(
                retention_reports(
                    (int(positive.sum()), int(negative.sum())), indices, values
                )
                if self.miner is not None
                else None
            ),
        )

    def _evaluate(self, prepared, collectives):
        """Execute the selected backend; all candidate forms share this call site."""
        ref_emb = (
            None if prepared.references is prepared.embeddings else prepared.references
        )
        validate_loss_inputs(
            self.input_contract,
            labels_present=prepared.labels is not None,
            tuple_present=prepared.indices is not None,
            ref_present=ref_emb is not None,
            location=f"MetricObjective.loss_fn_name={self.loss_fn_name}",
        )
        if prepared.native_wrapper:
            self.last_backend = "pml"
            scaled = pml_distributed.DistributedLossWrapper(
                self.base_loss, efficient=False
            )(prepared.embeddings, prepared.labels, indices_tuple=prepared.indices)
            reported = scaled.detach() / collectives.world_size
            return scaled + (reported - scaled.detach())
        self.last_backend = "autograd" if collectives.active else "local"
        value = evaluate_pml_loss(
            self.base_loss,
            prepared.embeddings,
            prepared.labels,
            indices_tuple=prepared.indices,
            ref_emb=ref_emb,
            pair_weights=prepared.pair_weights,
        )
        # Even an empty objective must traverse its differentiable gather.
        references = (
            prepared.embeddings if prepared.references is None else prepared.references
        )
        return value + references.sum() * 0

    def _input_signature(
        self, embeddings, labels, relations, pair_weights, indices_tuple, collectives
    ):
        """Validate local supervision before ranks agree on the execution mode."""
        validate_loss_inputs(
            self.input_contract,
            labels_present=labels is not None,
            tuple_present=indices_tuple is not None
            or relations is not None
            or self.miner is not None,
            location=f"MetricObjective.loss_fn_name={self.loss_fn_name}",
        )
        mode = validate_metric_inputs(embeddings, labels, relations, indices_tuple)
        if relations is not None and self.miner is not None:
            relation_miner_adapter(self.miner)
        if labels is None and not self.supports_relations:
            raise ValueError(
                f"MetricObjective.loss_fn_name={self.loss_fn_name}: requires class IDs; "
                "label-free tuple support is not declared"
            )
        pair_weight_arguments(self.base_loss, embeddings, None, pair_weights)
        if (
            relations is not None
            and collectives.active
            and self.distributed_backend == "pml"
        ):
            raise ValueError(
                "PML wrappers cannot gather prepared relations; "
                "use distributed_backend='auto' or 'autograd'"
            )
        if (
            pair_weights is not None
            and relations is None
            and indices_tuple is None
            and collectives.active
        ):
            raise ValueError(
                "Distributed labels-only pair_weights lack cross-rank coverage; "
                "provide explicit rank-local tuples or disable distributed candidates"
            )
        return (
            mode,
            embeddings.shape[1],
            str(embeddings.dtype),
            len(indices_tuple) if indices_tuple is not None else None,
            relations.exclusive_scenes if relations is not None else None,
            relations.source if relations is not None else None,
            pair_weights is not None,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        relations: RelationSupervision | None = None,
        pair_weights: torch.Tensor | None = None,
        indices_tuple: IndicesTuple | None = None,
        gather_fn: GatherFunction | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return scalar loss/reports for descriptors [A,D] and prepared supervision.

        Explicit tuples may accompany labels. With distributed execution their
        rank-local indices are offset into the gathered row space; no unobserved
        cross-rank pairs are added. Static relation candidates retain identity
        and unknown-edge constraints during preparation.

        pair_weights is finite, nonnegative [A,A] in the local descriptor order,
        for any supported supervision representation. Distributed labels-only
        weights are rejected: cross-rank candidate weights are not provided.
        Explicit tuples preserve local coverage; their weights are gathered as
        rank blocks. Native PML wrappers cannot transmit dynamic pair weights.
        gather_fn overrides any enclosing runtime binding; MetricModule normally
        supplies Lightning's public all_gather through the criterion call scope.

        unused_anchor_fraction reports the fraction of prepared rows absent from
        the positive-pair/triplet anchor indices. It is zero when no explicit
        tuple is prepared; it is not a classification or retrieval accuracy.
        """
        collectives = CandidateCollectives(self.distributed, gather_fn)
        error, signature = None, None
        try:
            signature = self._input_signature(
                embeddings, labels, relations, pair_weights, indices_tuple, collectives
            )
        except (TypeError, ValueError, IndexError, AttributeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        check_rank_consistency(error, signature, enabled=self.distributed)
        self.last_pair_shape = None
        with torch.autocast(embeddings.device.type, enabled=False):
            values = (
                embeddings.float()
                if embeddings.dtype in (torch.float16, torch.bfloat16)
                else embeddings
            )
            if self.normalize_embeddings:
                values = F.normalize(values, dim=-1, eps=1e-6)
            if relations is not None:
                prepared = self._prepare_relations(
                    values, relations, pair_weights, collectives
                )
            else:
                prepared = self._prepare_native(
                    values, labels, indices_tuple, pair_weights, collectives
                )
            main = self._evaluate(prepared, collectives)
            count = len(prepared.embeddings)
            if prepared.native_wrapper:
                count *= collectives.world_size
            unused = values.new_zeros(())
            if prepared.indices is not None:
                unused = values.new_tensor(
                    1 - len(torch.unique(prepared.indices[0])) / max(count, 1)
                )
            return {
                "loss": main,
                "unused_anchor_fraction": unused,
                **(prepared.mining_reports or {}),
            }


def validate_local_indices(indices, count, device):
    """Validate equal-sided PML pairs/triplets before rank offset remapping."""
    if len(indices) not in (3, 4):
        raise ValueError("Expected a pair or triplet indices tuple")
    for index in indices:
        if (
            index.ndim != 1
            or index.dtype not in (torch.int32, torch.int64)
            or (index < 0).any()
            or (index >= count).any()
        ):
            raise ValueError("Invalid pair/triplet indices")
    lengths = [len(index) for index in indices]
    if (len(indices) == 3 and len(set(lengths)) != 1) or (
        len(indices) == 4 and (lengths[0] != lengths[1] or lengths[2] != lengths[3])
    ):
        raise ValueError("PML index sides must have matching lengths")
    return tuple(index.to(device=device, dtype=torch.long) for index in indices)


def validate_metric_inputs(embeddings, labels, relations, indices_tuple):
    """Validate local target row spaces before ranks begin candidate collectives."""
    if embeddings.ndim != 2 or not embeddings.is_floating_point():
        raise ValueError("embeddings must be floating valid rows [M,D]")
    if (
        relations is not None and (labels is not None or indices_tuple is not None)
    ) or (relations is None and labels is None and indices_tuple is None):
        raise ValueError(
            "Supply prepared relations OR labels/indices_tuple (including their legal combination)"
        )
    count = len(embeddings)
    if labels is not None:
        labels = torch.as_tensor(labels)
        if (
            labels.ndim != 1
            or len(labels) != count
            or labels.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "Class labels must be integer [N], aligned with embeddings"
            )
        if indices_tuple is not None:
            validate_local_indices(indices_tuple, count, embeddings.device)
        return "labels+tuples" if indices_tuple is not None else "ids"
    if indices_tuple is not None:
        validate_local_indices(indices_tuple, count, embeddings.device)
        return "tuples"
    if not isinstance(relations, RelationSupervision):
        raise TypeError("relations must be data-prepared RelationSupervision")
    if len(relations.row_indices) != count:
        raise ValueError("Prepared relation rows must align with descriptors")
    for mask in (relations.positive, relations.negative, relations.known):
        if mask.shape != (count, count) or mask.dtype != torch.bool:
            raise ValueError("Prepared relation masks must be boolean [M,M]")
    if (relations.positive & relations.negative).any():
        raise ValueError("Prepared positive and negative masks must be disjoint")
    validate_relation_identities(relations)
    scenes = relations.scene_ids
    different_scenes = (
        (scenes[:, None] != scenes[None, :])
        & (scenes[:, None] >= 0)
        & (scenes[None, :] >= 0)
    )
    if relations.exclusive_scenes and (relations.positive & different_scenes).any():
        raise ValueError("Scene identities contradict positive GT")
    return "relations"
