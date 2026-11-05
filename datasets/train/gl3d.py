"""GL3D train reader composing the existing SupScene graph and sampling tools."""

import logging
from collections.abc import Sequence
from pathlib import Path

from gmlab.datasets.utils.supscene.scenegraph import SceneGraph
from gmlab.datasets.utils.supscene.subgraph_dataset import SubgraphDataset
from gmlab.datasets.utils.supscene.subgraph_sampler import SubgraphSampler


class GL3DSubgraphDataset(SubgraphDataset):
    """Scan an explicit scene root/list without loading any evaluation split.

    Missing/corrupt scene metadata is skipped with its reason as in the old
    reader. Explicit scene_names enable disjoint validation-scene holdouts.
    Training item/GT/sampling distributions retain SubgraphDataset's behavior.
    sampler accepts an external sample(graph, n_sub) strategy. With no strategy,
    use the existing anchor_expand defaults and iou_th=0.2 (or an explicit
    iou_th). Supplying both sampler and iou_th is an error, even if the latter
    equals 0.2, so no default-strategy setting is silently ignored.
    """

    def __init__(
        self,
        dataset_path,
        transform,
        *,
        scene_names=None,
        sampler=None,
        iou_th=None,
        n_sub=256,
        scenes_per_epoch=None,
        samples_per_scene=None,
        adaptive_sampling=True,
        min_images_per_scene=50,
        identity_namespace=None,
        sampling_multiplier=1.0,
    ):
        if scene_names is not None:
            if isinstance(scene_names, (str, bytes)) or not isinstance(
                scene_names, Sequence
            ):
                raise TypeError("scene_names must be a sequence of scene names")
            if any(not isinstance(name, str) or not name for name in scene_names):
                raise ValueError("scene_names must contain nonempty strings")
            scene_names = tuple(scene_names)
        if sampler is not None and iou_th is not None:
            raise ValueError("Provide sampler or iou_th, not both")
        if sampler is None:
            sampler = SubgraphSampler(iou_th=0.2 if iou_th is None else iou_th)
        self.dataset_path = Path(dataset_path)
        self.scene_names = scene_names
        super().__init__(
            n_sub=n_sub,
            transform=transform,
            sampler=sampler,
            scenes_per_epoch=scenes_per_epoch,
            samples_per_scene=samples_per_scene,
            adaptive_sampling=adaptive_sampling,
            min_images_per_scene=min_images_per_scene,
            # Stable source identity retained across the reader's module move.
            identity_namespace=(
                identity_namespace or "gmlab.datasets.train.gl3d.GL3DSubgraphDataset"
            ),
            sampling_multiplier=sampling_multiplier,
        )

    def _prepare_dataset(self):
        root = self.dataset_path
        if self.scene_names is not None:
            paths = [root / name for name in self.scene_names]
        elif (root / "gt.npz").is_file():
            paths = [root]
        else:
            paths = sorted(p for p in root.iterdir() if p.is_dir())
        self.scene_graphs = []
        for path in paths:
            try:
                graph = SceneGraph(str(path))
                if graph.N:
                    self.scene_graphs.append(graph)
            except (OSError, ValueError, KeyError) as error:
                logging.getLogger(__name__).warning("Skipping %s: %s", path, error)
