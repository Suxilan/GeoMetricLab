"""University1652 class/image discovery for the shared sample pipeline."""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from gmlab.datasets.utils.supscene.scenegraph import SceneGraph
from gmlab.datasets.utils.supscene.subgraph_dataset import SubgraphDataset
from gmlab.datasets.utils.supscene.subgraph_sampler import SubgraphSampler
from gmlab.datasets.utils.geo_dataset import GeoDataset
from gmlab.datasets.utils.image_list import load_rgb_image


class University1652InstanceDataset(GeoDataset):
    """Read one class's modality candidates, leaving selection to shared policy.

    One Dataset index addresses a class. Source-scoped physical image identities
    are independent of that index, class labels and repeated sampling exposures.
    Views and class subsets retain their explicit order/sorted local labels.
    Configure quotas explicitly to select sampled class views. With sampler=None
    the common base loads every candidate; no historical quota is implied.
    Sampling constraints and transforms belong to GeoDataset and its sampler;
    this reader has no modality-specific sampling rules.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = "u1652_instance",
        views: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        transform=None,
        sampler=None,
        *,
        source_scope: str | None = None,
        sampling_multiplier: float = 1.0,
    ) -> None:
        if views is not None and (
            isinstance(views, (str, bytes)) or not isinstance(views, Sequence)
        ):
            raise TypeError("views must be a sequence of modality names")
        if class_names is not None:
            if isinstance(class_names, (str, bytes)) or not isinstance(
                class_names, Sequence
            ):
                raise TypeError("class_names must be a sequence of class names")
            if any(not isinstance(name, str) or not name for name in class_names):
                raise ValueError("class_names must contain nonempty strings")
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.views = tuple(("satellite", "drone") if views is None else views)
        self.source_scope = source_scope if source_scope is not None else dataset_name
        super().__init__(
            modalities=self.views,
            transform=transform,
            sampler=sampler,
            identity_namespace=self.source_scope,
            sampling_multiplier=sampling_multiplier,
        )
        anchor_view = self.views[0]
        anchor_root = Path(dataset_path) / anchor_view
        self.classes = sorted(
            path.name for path in anchor_root.iterdir() if path.is_dir()
        )
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        if class_names is not None:
            if not set(class_names) <= self.class_to_idx.keys():
                raise ValueError("Unknown training class in explicit class selection")
            selected = set(class_names)
            self.classes = [name for name in self.classes if name in selected]
            self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

        self.samples = {view: {} for view in self.views}
        self.num_images = 0
        for view in self.views:
            for name in self.classes:
                directory = os.path.join(dataset_path, view, name)
                if not os.path.isdir(directory):
                    continue
                # Keep the source listing order: changing it changes fixed-seed
                # choices relative to the existing reader's image populations.
                paths = [
                    os.path.join(directory, filename)
                    for filename in os.listdir(directory)
                    if filename.lower().endswith((".png", ".jpg", ".jpeg"))
                    and os.path.isfile(os.path.join(directory, filename))
                ]
                self.samples[view][name] = paths
                self.num_images += len(paths)
        self.class_geo_ids = tuple(self.geo_id(name) for name in self.classes)
        self.num_classes = len(self.classes)
        self.num_scenes = self.num_classes

    def __len__(self) -> int:
        return self.num_classes

    def candidates(self, index: int) -> Mapping[str, Sequence[dict[str, Any]]]:
        """Return source rows without drawing samples or applying transforms."""
        name = self.classes[index]
        return {
            view: [
                {
                    "inputs": path,
                    "sample_id": (
                        self.source_scope,
                        str(Path(path).relative_to(self.dataset_path)),
                    ),
                    "geo_id": self.geo_id(name),
                    "modality": view,
                    "label": self.class_to_idx[name],
                }
                for path in self.samples[view].get(name, [])
            ]
            for view in self.views
        }

    def load_input(self, row: Mapping[str, Any]) -> Image.Image:
        """Decode an RGB image; the shared pipeline applies its transform."""
        return load_rgb_image(row)


class University1652SubgraphDataset(SubgraphDataset):
    """Read GT node order verbatim, with images relative to the training root.

    Missing sparse edges retain zero overlap. Scene names are building IDs;
    neither these IDs nor modality names are converted into class labels.
    Each graph declares node modalities using the first image-path component.
    n_sub bounds sampled nodes; the default 55 covers the current full scenes.
    Smaller selections do not guarantee inclusion of any particular modality.
    """

    def __init__(
        self,
        dataset_path,
        transform=None,
        *,
        gt_root=None,
        scene_names=None,
        views=("drone", "satellite"),
        n_sub=55,
        sampler=None,
        iou_th=None,
        scenes_per_epoch=None,
        samples_per_scene=None,
        adaptive_sampling=True,
        min_images_per_scene=1,
        identity_namespace="university1652",
        sampling_multiplier=1.0,
    ):
        self.dataset_path = Path(dataset_path)
        self.gt_root = (
            Path(gt_root) if gt_root is not None else self.dataset_path / "gt"
        )
        if scene_names is not None:
            if isinstance(scene_names, str) or not isinstance(
                scene_names, (list, tuple)
            ):
                raise TypeError("scene_names must be a sequence of scene names")
            if any(not isinstance(name, str) or not name for name in scene_names):
                raise ValueError("scene_names must contain nonempty strings")
        self.scene_names = scene_names
        if sampler is not None and iou_th is not None:
            raise ValueError("Provide sampler or iou_th, not both")
        if sampler is None:
            sampler = SubgraphSampler(iou_th=0.2 if iou_th is None else iou_th)
        super().__init__(
            n_sub=n_sub,
            sampler=sampler,
            transform=transform,
            scenes_per_epoch=scenes_per_epoch,
            samples_per_scene=samples_per_scene,
            adaptive_sampling=adaptive_sampling,
            min_images_per_scene=min_images_per_scene,
            identity_namespace=identity_namespace,
            modalities=views,
            sampling_multiplier=sampling_multiplier,
        )

    def _prepare_dataset(self):
        paths = (
            [self.gt_root / name for name in self.scene_names]
            if self.scene_names is not None
            else sorted(path for path in self.gt_root.iterdir() if path.is_dir())
        )
        self.scene_graphs = []
        for path in paths:
            graph = SceneGraph(str(path), image_root=str(self.dataset_path))
            modalities = []
            for name, image_path in zip(graph.image_names, graph.image_paths):
                relative = Path(name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or len(relative.parts) < 3
                ):
                    raise ValueError(
                        f"GT image must be relative to the training root: {name}"
                    )
                modality, scene = relative.parts[:2]
                if modality not in self.declared_modalities or scene != path.name:
                    raise ValueError(
                        f"GT node has inconsistent modality or scene: {name}"
                    )
                if not Path(image_path).is_file():
                    raise FileNotFoundError(image_path)
                modalities.append(modality)
            graph.modalities = tuple(modalities)
            self.scene_graphs.append(graph)
