"""SupScene DataModule for PyTorch Lightning"""

from __future__ import annotations

from typing import Any, Dict, Optional
import pprint
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
except Exception as exc:
    raise ImportError(
        "pytorch-lightning is required for the DataModule. Install via `pip install pytorch-lightning`."
    ) from exc

from config.transform_hubs import get_transform
from src.supscene.datasets.collate import make_pad_collate
from src.utils.logger import print_rank_0


class SupSceneDataModule(pl.LightningDataModule):
    """DataModule for SupScene subgraph training.
    
    Args:
        train_datapath: Root path for training dataset.
        batch_size: Global batch size.
        num_workers: DataLoader workers.
        n_sub: Number of nodes per subgraph.
        transform: Image transforms.
        scenes_per_epoch: Number of scenes per epoch (None = all scenes).
        samples_per_scene: Fixed samples per scene (None = adaptive).
        adaptive_sampling: Enable adaptive per-scene sampling.
        min_images_per_scene: Filter scenes with fewer images.
        collate_diag_weight: Optional diagonal weight for collate function.
    """

    def __init__(
        self,
        train_datapath: Optional[str] = None,
        train_set_name: Optional[str] = None,
        train_image_size: tuple[int, int] = (224, 224),
        batch_size: int = 1,
        val_batch_size: int = 128,
        num_workers: int = 4,
        n_sub: int = 128,
        iou_th: float = 0.2,
        scenes_per_epoch: Optional[int] = None,
        samples_per_scene: Optional[int] = None,
        adaptive_sampling: bool = True,
        min_images_per_scene: int = 50,
        collate_diag_weight: Optional[float] = None,
        batch_sampler: Optional[Any] = None,
        val_set_names: Optional[list[str]] = None,
        val_image_size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        
        if train_set_name is None:
            raise ValueError("Either 'train_set_name' must be provided")
        
        self.train_datapath = train_datapath
        self.train_set_name = train_set_name
        self.train_image_size = train_image_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.n_sub = n_sub
        self.iou_th = iou_th
        self.scenes_per_epoch = scenes_per_epoch
        self.samples_per_scene = samples_per_scene
        self.adaptive_sampling = adaptive_sampling
        self.min_images_per_scene = min_images_per_scene
        self.collate_diag_weight = collate_diag_weight
        self.batch_sampler = batch_sampler
        self.val_set_names = val_set_names if val_set_names is not None else []
        self.val_image_size = val_image_size or train_image_size

        self.extra_data_config = {
            "n_sub": n_sub,
            "iou_th": iou_th,
            "scenes_per_epoch": scenes_per_epoch,
            "samples_per_scene": samples_per_scene,
            "adaptive_sampling": adaptive_sampling,
            "min_images_per_scene": min_images_per_scene,
            "collate_diag_weight": collate_diag_weight,
            "batch_sampler": batch_sampler,
        }

        print_rank_0(f"[{self.__class__.__name__}] Initialized with train_set={self.train_set_name}, val_sets={self.val_set_names}")

        self.train_dataset = None
        self.val_datasets = None 
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training/validation."""
        if stage in (None, "fit", "validate", "train"):
            self.train_dataset = self._get_train_dataset(self.train_set_name)
            self.val_datasets = [self._get_val_dataset(ds_name) for ds_name in self.val_set_names]
   
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        collate_fn = make_pad_collate(diag_weight=self.collate_diag_weight)
        
        if self.batch_sampler is not None:
            return DataLoader(
                dataset=self.train_dataset,
                num_workers=self.num_workers,
                batch_sampler=self.batch_sampler,
                pin_memory=True,
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_fn,
        )
    
    def val_dataloader(self) -> list[DataLoader]:
        val_dataloaders = []
        for i,dataset in enumerate(self.val_datasets):
            if dataset.__class__.__name__ == "GL3DSceneDataset":
                from src.datasets.valid.gl3d import SceneBatchSampler
                sampler = SceneBatchSampler(
                    dataset,
                    batch_size=self.val_batch_size,
                    drop_last=False,
                    shuffle_scenes=False,
                    shuffle_within_scene=False,
                )
                dl = DataLoader(
                    dataset=dataset,
                    num_workers=self.num_workers,
                    batch_sampler=sampler,
                    pin_memory=True,
                )
            else:
                dl = DataLoader(
                        dataset=dataset,
                        # batch_size=self.batch_size,
                        batch_size=self.val_batch_size,
                        num_workers=self.num_workers,
                        drop_last=False,
                        pin_memory=True,
                        shuffle=False
                    )
            val_dataloaders.append(dl)
        return val_dataloaders

    def _get_train_dataset(self, ds_name) -> Any:
        """Build dataset for training."""
        from src.datasets import GL3DSubgraphDataset

        ds = ds_name or getattr(self, "train_set_name", "gl3d_subgraph")
        ds_key = ds.lower().split("_")[0]

        if ds.lower() == "gl3d_subgraph":
            return GL3DSubgraphDataset(
                dataset_name=ds,
                dataset_path=self.train_datapath,
                input_transform=get_transform(ds_key, self.train_image_size, train=True),
                n_sub=self.n_sub,
                iou_th=self.iou_th,
                scenes_per_epoch=self.scenes_per_epoch,
                samples_per_scene=self.samples_per_scene,
                adaptive_sampling=self.adaptive_sampling,
                min_images_per_scene=self.min_images_per_scene,
            )

        raise ValueError(f"Unsupported train dataset: {ds}")

    def _get_val_dataset(self, ds_name) -> Any:
        # Build dataset for validation/evaluation
        from src.datasets import GL3DDataset, GL3DSceneDataset, University1652Dataset

        ds = ds_name or getattr(self, "train_set_name", "gl3d")
        ds_key = ds.lower().split("_")[0]

        if "gl3d" in ds.lower():
            if "scene" in ds.lower():
                return GL3DSceneDataset(
                    dataset_name=ds,
                    input_transform=get_transform(ds_key, self.val_image_size, train=False),
                )
            return GL3DDataset(
                dataset_name=ds,
                input_transform=get_transform(ds_key, self.val_image_size, train=False),
            )
        elif "university1652" in ds.lower() or "u1652" in ds.lower():
            return University1652Dataset(
                dataset_name=ds,
                input_transform=get_transform("u1652", self.val_image_size, train=False),
            )
        raise ValueError(f"Unsupported val dataset: {ds}")