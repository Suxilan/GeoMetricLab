"""Training candidate collection; generic communication stays in utils."""

import torch

from ..losses.distributed import MetricCollectives
from .candidates import RelationBatch


class CandidateCollectives(MetricCollectives):
    """Gather aligned candidate tensors without interpreting raw annotations."""

    def global_relations(self, batch: RelationBatch) -> RelationBatch:
        """Gather descriptors and preserve observed GT as diagonal rank blocks."""
        sizes = self.sizes(len(batch.embeddings), batch.embeddings.device)
        group_counts = self.sizes(batch.num_groups, batch.embeddings.device)
        embeddings = torch.cat(
            self.gather_variable(batch.embeddings, sizes, sync_grads=True)
        )
        matrices = [
            torch.block_diag(*self.gather_variable(matrix, sizes, square=True))
            for matrix in (
                batch.positive,
                batch.negative,
                batch.pair_weights,
                batch.known,
            )
        ]
        groups = self.gather_variable(batch.groups, sizes)
        offset = 0
        for index, count in enumerate(group_counts):
            groups[index] = groups[index] + offset
            offset += count
        return RelationBatch(
            embeddings,
            *matrices,
            torch.cat(groups),
            torch.cat(self.gather_variable(batch.scene_ids, sizes)),
            torch.cat(self.gather_variable(batch.sample_ids, sizes)),
            sum(group_counts),
        )

    def ids(
        self, embeddings: torch.Tensor, labels: torch.Tensor, labels_scope: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather [N,D] descriptors and [N] IDs; rank-local classes stay distinct."""
        if labels_scope not in {"global", "rank"}:
            raise ValueError("labels_scope must be global or rank")
        sizes = self.sizes(len(embeddings), embeddings.device)
        embeddings = torch.cat(self.gather_variable(embeddings, sizes, sync_grads=True))
        parts = self.gather_variable(labels, sizes)
        if labels_scope == "global":
            return embeddings, torch.cat(parts)
        keys = [
            torch.stack((torch.full_like(part, rank), part), dim=1)
            for rank, part in enumerate(parts)
        ]
        labels = torch.unique(torch.cat(keys), dim=0, return_inverse=True)[1]
        return embeddings, labels
