"""Continuous positive-pair weighting through PML's native loss lifecycle."""

import torch
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from ..contracts import LossInputContract


class WeightedSupConLoss(SupConLoss):
    """SupCon with nonnegative positive weights [anchors, references].

    Pair weights normalize the positive log-probability mean per anchor; they
    do not weight negative candidates in the denominator. Without weights the
    native SupCon implementation is used. Weighted singleton/all-positive cases
    retain the former implementation's math. PML owns input handling, distance,
    reducer, statistics and embedding regularization.
    """

    supports_pair_weights = True
    input_contract = LossInputContract(
        requires_labels=False, tuple_usage="pairs", supports_ref_emb=True
    )

    def __init__(self, positive_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.positive_gamma = float(positive_gamma)
        self._pair_weights = None

    def make_pair_weights(self, overlap):
        return overlap.clamp_min(0).pow(self.positive_gamma)

    def forward(
        self,
        embeddings,
        labels=None,
        indices_tuple=None,
        ref_emb=None,
        ref_labels=None,
        pair_weights=None,
    ):
        if pair_weights is not None:
            references = embeddings if ref_emb is None else ref_emb
            if (
                not torch.is_tensor(pair_weights)
                or not pair_weights.is_floating_point()
                or pair_weights.shape != (len(embeddings), len(references))
                or not torch.isfinite(pair_weights).all()
                or (pair_weights < 0).any()
            ):
                raise ValueError(
                    "pair_weights must be finite nonnegative floating [A,R]"
                )
        previous = self._pair_weights
        self._pair_weights = pair_weights
        try:
            return super().forward(
                embeddings, labels, indices_tuple, ref_emb, ref_labels
            )
        finally:
            self._pair_weights = previous

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        if self._pair_weights is None:
            return super().compute_loss(
                embeddings, labels, indices_tuple, ref_emb, ref_labels
            )
        # Native GenericPairLoss drops singleton tuples before computing a
        # matrix. The established weighted objective explicitly retains them.
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        pairs = lmu.convert_to_pairs(indices_tuple, labels, ref_labels)
        return self.loss_method(self.distance(embeddings, ref_emb), pairs)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        if self._pair_weights is None:
            return super()._compute_loss(mat, pos_mask, neg_mask)
        logits = (mat if self.distance.is_inverted else -mat) / self.temperature
        mask = (pos_mask + neg_mask).bool()
        safe = logits.masked_fill(~mask, -torch.inf)
        safe = torch.where(mask.any(1, keepdim=True), safe, torch.zeros_like(safe))
        log_probability = (
            logits - torch.logsumexp(safe, dim=1, keepdim=True)
        ).masked_fill(~mask, 0)
        weights = self._pair_weights.to(logits).masked_fill(~pos_mask.bool(), 0)
        values = -(weights * log_probability).sum(1) / (weights.sum(1) + 1e-8)
        return {
            "loss": {
                "losses": values,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }
