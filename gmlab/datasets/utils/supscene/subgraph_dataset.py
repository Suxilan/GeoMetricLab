"""Shared scene exposure, node sampling and image/GT composition."""

import os
import math
import random
from abc import ABC, abstractmethod
from numbers import Integral
from typing import Dict, Optional, Any, List
import torch
import numpy as np
import logging
from .subgraph_sampler import SubgraphSampler
from ....components import build_component
from ..identities import integer_tensor
from ..geo_dataset import GeoDataset
from ..image_list import load_rgb_image
from ..relations import stable_id


def _scene_count(value, name, *, minimum=1):
    """Reject lossy count conversions at the scene-plan boundary."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


class SubgraphDataset(GeoDataset, ABC):
    """Reusable scene-plan, node sampling, static GT and image composition.

    A reader may implement _prepare_dataset to populate scene_graphs, or callers
    can inject graphs with SceneSubgraphDataset. Graph-specific selection and
    group GT use GeoDataset.prepare_rows for selected input preparation.

    Public requests are an integer epoch-plan position in [0, len(self)), or
    ('scene', index) addressing the *filtered* scenes list in [0, num_scenes).
    The latter is the common GeoEpochSampler/worker contract. Negative,
    boolean, malformed and out-of-range requests are rejected, not wrapped.
    """

    def __init__(
        self,
        n_sub: int = 256,
        sampler: Any = None,
        transform: Optional[Any] = None,
        scenes_per_epoch: Optional[int] = None,
        samples_per_scene: Optional[int] = None,
        adaptive_sampling: bool = True,
        min_images_per_scene: int = 50,
        identity_namespace: str | None = None,
        modalities=("image",),
        sampling_multiplier: float = 1.0,
    ):
        """Compose scene loading, node selection and an epoch exposure plan.

        n_sub bounds nodes per item. sampler.sample(graph, n_sub) returns NumPy
        integer node indices; None selects the existing uniform policy. Readers
        populate scene_graphs before filtering by min_images_per_scene.
        samples_per_scene fixes exposure; when None with adaptive_sampling=True,
        the existing coverage policy determines each scene's exposure count.
        transform maps RGB PIL images to float [C,H,W] tensors."""
        super().__init__(
            modalities=modalities,
            transform=transform,
            identity_namespace=identity_namespace,
            sampling_multiplier=sampling_multiplier,
        )
        self.n_sub = _scene_count(n_sub, "n_sub")
        self.adaptive_sampling = adaptive_sampling
        self.min_images_per_scene = _scene_count(
            min_images_per_scene, "min_images_per_scene", minimum=0
        )

        self.sampler = (
            SubgraphSampler(mode="uniform")
            if sampler is None
            else build_component(sampler)
        )
        if not callable(getattr(self.sampler, "sample", None)):
            raise TypeError("sampler must provide sample(graph, n_sub)")

        self.scene_graphs: List[Any] = []
        self.scenes: List[Any] = []

        self._prepare_dataset()

        if self.min_images_per_scene > 0:
            self.scenes = [
                s for s in self.scene_graphs if s.N >= self.min_images_per_scene
            ]
        else:
            self.scenes = self.scene_graphs

        if len(self.scenes) == 0:
            raise ValueError(
                f"No valid scenes found: min_images_per_scene={self.min_images_per_scene}. "
                f"Total scenes={len(self.scene_graphs)}. Lower the threshold or check dataset."
            )

        self.num_scenes = len(self.scenes)
        self.scenes_per_epoch = (
            _scene_count(scenes_per_epoch, "scenes_per_epoch")
            if scenes_per_epoch is not None
            else self.num_scenes
        )

        if samples_per_scene is None and adaptive_sampling:
            self.samples_per_scene_list = [
                self._adaptive_samples_per_scene(scene.N) for scene in self.scenes
            ]
            self.samples_per_scene = 1  # unused in adaptive mode
        else:
            self.samples_per_scene = (
                1
                if samples_per_scene is None
                else _scene_count(samples_per_scene, "samples_per_scene")
            )
            self.samples_per_scene_list = None

        self._build_epoch_indices()

    def print(self, message: str):
        logging.getLogger(__name__).info("[%s] %s", self.__class__.__name__, message)

    @abstractmethod
    def _prepare_dataset(self):
        """Populate scene_graphs from this reader's static source metadata."""
        pass

    def _adaptive_samples_per_scene(
        self,
        image_count: int,
        target_coverage: float = 0.20,
        min_samples: int = 2,
        max_samples: int = 32,
        mid_scene_boost: float = 1.5,
    ) -> int:
        """Hybrid scene-level sampling: linear coverage + sqrt smooth.

        - Small scenes: at least *min_samples* (never ignored).
        - Mid scenes: ``mid_scene_boost * sqrt(N/n_sub)`` lifts them
          above the linear floor so they aren't all clamped to *min_samples*.
        - Large scenes: ``target_coverage * N / n_sub`` ensures they get
          proportional representation, capped by *max_samples*.
        """
        image_count = int(image_count)
        if image_count <= 0:
            return int(min_samples)

        coverage_ratio = image_count / float(self.n_sub)
        linear_exposure = target_coverage * coverage_ratio
        sqrt_exposure = mid_scene_boost * np.sqrt(coverage_ratio)

        samples = int(np.ceil(max(linear_exposure, sqrt_exposure)))
        samples = max(int(min_samples), samples)
        return int(min(samples, int(max_samples)))

    @property
    def epoch_size(self) -> int:
        """Reject variable-length partial scene populations before loader setup."""
        exposures = self.samples_per_scene_list
        if (
            exposures is not None
            and self.scenes_per_epoch < self.num_scenes
            and len(set(exposures)) > 1
        ):
            raise ValueError(
                "Scene sampling requires a fixed epoch budget: adaptive sampling "
                "with unequal scene exposures cannot select only part of the "
                "scenes. Set samples_per_scene explicitly or use all scenes."
            )
        return super().epoch_size

    def _scene_round(self, rng, shuffle=True):
        if self.samples_per_scene_list is not None:
            population = list(range(self.num_scenes))
            count = min(self.scenes_per_epoch, self.num_scenes)
            pool = rng.sample(population, k=count) if shuffle else population[:count]
            indices = [i for i in pool for _ in range(self.samples_per_scene_list[i])]
        else:
            count = self.scenes_per_epoch * self.samples_per_scene
            repeats = math.ceil(count / self.num_scenes)
            indices = (list(range(self.num_scenes)) * repeats)[:count]
        if shuffle:
            rng.shuffle(indices)
        return indices

    def _epoch_round(self, rng, shuffle):
        """Return stable scene requests; workers sample nodes independently."""
        return [("scene", index) for index in self._scene_round(rng, shuffle)]

    def _build_epoch_indices(self) -> None:
        """Keep the direct-reader epoch API using its caller-owned RNG."""
        self.epoch_scene_indices = self._scene_round(random)
        self.total_samples = len(self.epoch_scene_indices)

    def reshuffle_epoch(
        self,
        epoch: Optional[int] = None,
        scenes_per_epoch: Optional[int] = None,
        samples_per_scene: Optional[int] = None,
    ) -> None:
        """Rebuild the scene plan at an epoch boundary.

        epoch is an accepted scheduler hook argument; the caller owns RNG seeding.
        Explicit samples_per_scene disables adaptive counts. Worker requests use
        stable scene indices and do not read a stale copy of this mutable plan."""
        next_scenes = (
            self.scenes_per_epoch
            if scenes_per_epoch is None
            else _scene_count(scenes_per_epoch, "scenes_per_epoch")
        )
        next_samples = (
            self.samples_per_scene
            if samples_per_scene is None
            else _scene_count(samples_per_scene, "samples_per_scene")
        )
        self.scenes_per_epoch = next_scenes
        self.samples_per_scene = next_samples
        if samples_per_scene is not None:
            self.samples_per_scene_list = None
        self._build_epoch_indices()

    def _get_scene_graph(self, scene_idx: int):
        """Return a scene by its index in the filtered scene list."""
        return self.scenes[scene_idx]

    def _scene_index(self, request):
        if isinstance(request, tuple):
            if (
                len(request) != 2
                or not isinstance(request[0], str)
                or request[0] != "scene"
            ):
                raise ValueError("Explicit scene request must be ('scene', index)")
            index, limit, kind = request[1], self.num_scenes, "scene"
        else:
            index, limit, kind = request, len(self.epoch_scene_indices), "epoch"
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError(f"{kind} index must be an integer")
        if not 0 <= index < limit:
            raise IndexError(f"{kind} index {index} outside [0, {limit})")
        return int(index) if kind == "scene" else self.epoch_scene_indices[int(index)]

    def _sample_subgraph(self, request):
        graph = self._get_scene_graph(self._scene_index(request))
        node_idx = integer_tensor(
            self.sampler.sample(graph, self.n_sub),
            name="Sampled node indices",
            device="cpu",
            minimum=0,
            maximum=graph.N - 1,
        )
        if node_idx.ndim != 1:
            raise ValueError("Sampled node indices must be a vector")
        return node_idx.numpy(), graph

    def load_input(self, row):
        """Decode a selected image; GeoDataset owns transform application."""
        return load_rgb_image(row)

    def relation_resolver(self, *, pos_th=0.3, neg_th=None):
        """Build a read-only GT query for selected local or gathered identities.

        The query owns references to static graphs, not Dataset workers or
        training state. Thresholds must match the loader's supervision decoder.
        """
        from .relations import SceneRelations

        graphs = {
            stable_id(self.geo_id(os.path.basename(graph.scene_dir))): graph
            for graph in self.scenes
        }
        if len(graphs) != len(self.scenes):
            raise ValueError("Duplicate scene keys in one identity namespace")
        return SceneRelations(
            graphs, pos_th=pos_th, neg_th=neg_th, source=self.identity_namespace
        )

    def node_modality(self, graph, node_index: int) -> str:
        """Return a node's declared modality; readers may provide aligned metadata."""
        names = getattr(graph, "modalities", None)
        return "image" if names is None else names[node_index]

    def __getitem__(self, request: int | tuple[str, int]) -> Dict:
        """Read one subgraph by epoch-plan position or explicit ("scene", index).

        Returns scene_id (display name), scene_key (canonical source identity),
        node_idx LongTensor [N], overlap FloatTensor [N,N], image_paths, images
        FloatTensor [N,3,H,W] or None, and num_nodes. sample_ids, modalities and
        row_positions describe the same selected-row order. Rows remain aligned across
        images, physical-image identities and static supervision."""
        node_indices, scene_graph = self._sample_subgraph(request)
        node_count = int(node_indices.shape[0])

        overlap = scene_graph.dense_overlap(node_indices, add_self=True)  # (n, n)

        paths = [scene_graph.image_paths[i] for i in node_indices.tolist()]

        scene_key = self.geo_id(os.path.basename(scene_graph.scene_dir))
        rows = self.prepare_rows(
            (
                {
                    "inputs": path,
                    "sample_id": (scene_key, int(node_index)),
                    "modality": self.node_modality(scene_graph, int(node_index)),
                }
                for path, node_index in zip(paths, node_indices.tolist())
            ),
            load_inputs=self.transform is not None,
        )
        images = (
            torch.stack([row["inputs"] for row in rows], dim=0)
            if self.transform is not None
            else None
        )

        return {
            "gt_source": self.identity_namespace,
            "exclusive_scenes": True,
            "scene_id": os.path.basename(scene_graph.scene_dir),
            "scene_key": scene_key,
            "sample_ids": [row["sample_id"] for row in rows],
            "modalities": [row["modality"] for row in rows],
            "row_positions": torch.tensor(
                [row["row_position"] for row in rows], dtype=torch.long
            ),
            "node_idx": torch.from_numpy(node_indices).long(),
            "overlap": torch.from_numpy(overlap),
            "image_paths": paths,
            "images": images,
            "num_nodes": node_count,
        }

    def __len__(self):
        return self.total_samples


class SceneSubgraphDataset(SubgraphDataset):
    """Compose externally supplied SceneGraph objects without a source reader.

    Use the same public integer/explicit-scene requests as SubgraphDataset.
    The injected sampler must implement sample(graph, n_sub) -> int64 NumPy
    node indices [n]; no inheritance or strategy registry is required. Omitting
    it retains the existing uniform-mode configuration (including its small-
    graph behavior). transform converts RGB PIL inputs to [C,H,W] tensors.

    Items contain node_idx [n], overlap [n,n], optional images [n,C,H,W], paths
    and canonical scene identity. Known missing graph edges stay zero-overlap
    negatives; cross-subgraph relations are not synthesized. Callers own scene
    exclusivity and epoch/worker scheduling, not Trainer or online candidates.
    """

    def __init__(self, scene_graphs, **kwargs):
        self._graphs = list(scene_graphs)
        super().__init__(**kwargs)

    def _prepare_dataset(self):
        self.scene_graphs = self._graphs
