"""Node-selection policies over static scene overlap graphs."""

import random
import math
from numbers import Integral
from typing import List, Dict
import numpy as np

from .scenegraph import SceneGraph


class SubgraphSampler:
    """Select static image nodes using uniform, anchor-expand or balanced sampling.

    anchor_expand uses overlap-threshold BFS from a random eligible anchor.
    seed_modality optionally restricts anchors using aligned graph.modalities;
    it never restricts the modalities of neighbors. balanced greedily targets
    a positive-pair ratio. Neither policy reads model
    embeddings, changes annotation semantics or manages distributed candidates."""

    def __init__(
        self,
        mode: str = "anchor_expand",
        iou_th: float = 0.0,
        topk_per_hop: int = 32,
        max_hops: int = 3,
        balanced_small_graph: bool = True,
        small_graph_threshold: int = 16,
        target_positive_ratio: float = 0.5,
        seed_modality: str | None = None,
    ):
        if mode not in {"uniform", "anchor_expand", "balanced"}:
            raise ValueError(f"Unknown subgraph sampling mode: {mode}")
        for name, value in (
            ("topk_per_hop", topk_per_hop),
            ("max_hops", max_hops),
            ("small_graph_threshold", small_graph_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not math.isfinite(iou_th):
            raise ValueError("iou_th must be finite")
        if not 0 <= target_positive_ratio <= 1:
            raise ValueError("target_positive_ratio must be in [0,1]")
        if type(balanced_small_graph) is not bool:
            raise TypeError("balanced_small_graph must be bool")
        if seed_modality is not None:
            if not isinstance(seed_modality, str) or not seed_modality:
                raise ValueError("seed_modality must be a nonempty modality name")
            if mode != "anchor_expand":
                raise ValueError("seed_modality requires anchor_expand sampling")
        self.seed_modality = seed_modality
        self.mode = mode
        self.iou_th = iou_th
        self.topk_per_hop = topk_per_hop
        self.max_hops = max_hops
        self.balanced_small_graph = balanced_small_graph
        self.small_graph_threshold = small_graph_threshold
        self.target_positive_ratio = target_positive_ratio

    def sample(self, graph: SceneGraph, n_sub: int) -> np.ndarray:
        """Return up to n_sub scene-local node indices as NumPy int64 [N].

        The scene graph supplies image count and static overlaps. Existing small-
        scene, expansion, ordering and RNG behavior are preserved."""
        if isinstance(n_sub, bool) or not isinstance(n_sub, Integral) or n_sub < 1:
            raise ValueError("n_sub must be a positive integer")
        node_count = graph.N
        seed_candidates = self._seed_candidates(graph)
        if n_sub >= node_count:
            return np.arange(node_count, dtype=np.int64)

        if (
            self.balanced_small_graph
            and n_sub <= self.small_graph_threshold
            and self.mode in ["uniform", "balanced"]
        ):
            return self._balanced_sample(graph, n_sub)

        if self.mode == "uniform":
            return np.random.choice(node_count, size=n_sub, replace=False)
        elif self.mode == "balanced":
            return self._balanced_sample(graph, n_sub)

        # anchor expansion based on adjacency
        # Annotation record multiplicity/order is part of the established
        # exposure protocol; canonical GT still uses one max per physical edge.
        adjacency = graph.neighbor_lists(self.iou_th, preserve_multiplicity=True)
        anchor = (
            np.random.randint(0, node_count)
            if seed_candidates is None
            else np.random.choice(seed_candidates)
        )
        visited = set([int(anchor)])
        frontier = [int(anchor)]
        hops = 0
        while len(visited) < n_sub and len(frontier) > 0 and hops < self.max_hops:
            new_frontier = []
            for source in frontier:
                neighbors = adjacency[source]
                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.topk_per_hop:
                    neighbors = random.sample(neighbors, self.topk_per_hop)
                for target in neighbors:
                    if target not in visited:
                        visited.add(int(target))
                        new_frontier.append(int(target))
                        if len(visited) >= n_sub:
                            break
                if len(visited) >= n_sub:
                    break
            frontier = new_frontier
            hops += 1

        if len(visited) < n_sub:
            remaining_nodes = [x for x in range(node_count) if x not in visited]
            additional_nodes = np.random.choice(
                remaining_nodes, size=(n_sub - len(visited)), replace=False
            )
            visited.update(additional_nodes.tolist())

        indices = np.fromiter(visited, dtype=np.int64)
        np.random.shuffle(indices)
        return indices[:n_sub]

    def _seed_candidates(self, graph):
        """Select seed eligibility from node metadata, never dataset names.

        Missing requested modalities fail rather than silently falling back to
        random seeds. The chosen seed remains in the sampled subgraph, although
        final row order is shuffled independently of expansion order.
        """
        if self.seed_modality is None:
            return None
        modalities = getattr(graph, "modalities", None)
        if modalities is None or len(modalities) != graph.N:
            raise ValueError(
                "Modality-seeded sampling requires aligned node modalities"
            )
        candidates = [
            index for index, name in enumerate(modalities) if name == self.seed_modality
        ]
        if not candidates:
            raise ValueError(f"Scene has no seed modality {self.seed_modality!r}")
        return candidates

    def _balanced_sample(self, graph: SceneGraph, n_sub: int) -> np.ndarray:
        """Greedily select nodes toward the configured positive/negative pair ratio."""
        node_count = graph.N
        if n_sub >= node_count:
            return np.arange(node_count, dtype=np.int64)

        # Cache static overlap lookups for repeated greedy candidate comparisons.
        positive_overlaps: Dict[tuple, float] = {}
        for source, target, overlap in zip(graph.row, graph.col, graph.val):
            if overlap >= self.iou_th:
                positive_overlaps[(source, target)] = overlap
                positive_overlaps[(target, source)] = overlap

        def has_overlap(i: int, j: int) -> bool:
            return (i, j) in positive_overlaps

        def positive_ratio(nodes: List[int]) -> float:
            if len(nodes) < 2:
                return 0.0
            positive_count, pair_count = 0, 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    pair_count += 1
                    if has_overlap(nodes[i], nodes[j]):
                        positive_count += 1
            return positive_count / pair_count if pair_count > 0 else 0.0

        nodes = np.random.choice(node_count, size=n_sub, replace=False).tolist()
        current_ratio = positive_ratio(nodes)

        max_iterations = min(50, node_count)  # max iterations
        tolerance = 0.05
        for _ in range(max_iterations):
            if abs(current_ratio - self.target_positive_ratio) <= tolerance:
                break
            need_more_positive = current_ratio < self.target_positive_ratio
            unused = [i for i in range(node_count) if i not in nodes]
            if not unused:
                break

            best_improvement = 0.0
            best_swap = None
            candidate_limit = min(20, len(unused))
            for replace_index in range(len(nodes)):
                for candidate in unused[:candidate_limit]:
                    trial = nodes.copy()
                    trial[replace_index] = candidate
                    candidate_ratio = positive_ratio(trial)
                    improvement = (
                        (candidate_ratio - current_ratio)
                        if need_more_positive
                        else (current_ratio - candidate_ratio)
                    )
                    if improvement > best_improvement:
                        best_improvement, best_swap = (
                            improvement,
                            (
                                replace_index,
                                candidate,
                            ),
                        )

            if best_swap and best_improvement > 1e-2:
                replace_index, candidate = best_swap
                nodes[replace_index] = candidate
                current_ratio = positive_ratio(nodes)
            else:
                break

        indices = np.array(nodes, dtype=np.int64)
        np.random.shuffle(indices)
        return indices
