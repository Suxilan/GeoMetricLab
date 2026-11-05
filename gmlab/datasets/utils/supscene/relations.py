"""Query static scene GT for an arbitrary selected candidate set."""

import math

import torch


class SceneRelations:
    """Resolve [M] scene/node IDs to [M,M] positive and negative masks.

    Missing edges inside a scene retain the graph's zero-overlap semantics.
    Different scenes are left to the caller's mutual-exclusion contract.
    Repeated physical nodes are excluded, including across rank boundaries.
    """

    def __init__(self, graphs, *, pos_th=0.3, neg_th=None, source=None):
        if not math.isfinite(pos_th) or (
            neg_th is not None and (not math.isfinite(neg_th) or neg_th >= pos_th)
        ):
            raise ValueError("Relation thresholds must be finite and ordered")
        self.key = None if source is None else (source, pos_th, neg_th)
        self.graphs = dict(graphs)
        self.pos_th = pos_th
        self.neg_th = neg_th

    def __call__(self, scene_ids, sample_ids):
        if scene_ids.ndim != 1 or sample_ids.shape != scene_ids.shape:
            raise ValueError("Scene and sample identities must be aligned vectors")
        scenes = scene_ids.detach().cpu()
        nodes = sample_ids.detach().cpu()
        positive = torch.zeros((len(scenes), len(scenes)), dtype=torch.bool)
        negative = torch.zeros_like(positive)
        for identity in scenes.unique().tolist():
            if identity not in self.graphs:
                raise ValueError(f"No GT source registered for scene {identity}")
            rows = torch.where(scenes == identity)[0]
            selected = nodes[rows]
            scores = torch.from_numpy(
                self.graphs[identity].dense_overlap(selected.numpy(), add_self=False)
            )
            distinct = selected[:, None] != selected[None, :]
            positive[rows[:, None], rows[None, :]] = (scores >= self.pos_th) & distinct
            negatives = (
                scores < self.pos_th if self.neg_th is None else scores <= self.neg_th
            )
            negative[rows[:, None], rows[None, :]] = negatives & distinct
        return positive.to(scene_ids.device), negative.to(scene_ids.device)
