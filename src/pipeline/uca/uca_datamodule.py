"""UCA DataModule for University1652."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from src.datasets import University1652UCADataset
from src.datasets.valid.u1652 import U1652_TASKS
from src.utils.io import load_features_h5

try:
    from src.rerank.faiss_utils import faiss_knn_topk
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class U1652FeatureValDataset(Dataset):
    def __init__(self, feature_file: Path, max_k: int = 100, dataset_root: str | Path | None = None):
        self.feature_file = Path(feature_file)
        if not self.feature_file.is_file():
            raise FileNotFoundError(f"Feature file not found: {self.feature_file}")

        # IMPORTANT: feature_file is a workspace-relative path; use cache_dir='.'
        data = load_features_h5(self.feature_file, cache_dir=Path("."))
        self.features = torch.from_numpy(data["features"]).float()
        md = data.get("metadata", {})

        self.num_references = int(md.get("num_references"))
        self.num_queries = int(md.get("num_queries"))

        m = re.search(r"(u1652_[a-z0-9]+2[a-z0-9]+)", self.feature_file.name.lower())
        if not m:
            raise ValueError(f"Cannot infer u1652 task name from feature file: {self.feature_file.name}")

        self.task_name = m.group(1)

        # Make dataset_name UNIQUE per feature file to avoid metric key overwrites.
        # e.g. 'convnext_..._u1652_drone2satellite_features' -> 'convnext_..._u1652_drone2satellite'
        stem = self.feature_file.stem
        if stem.endswith("_features"):
            stem = stem[: -len("_features")]
        self.dataset_name = stem

        if dataset_root is None:
            dataset_root = Path("data/university1652/test")
        else:
            dataset_root = Path(dataset_root)

        if self.task_name in U1652_TASKS:
            meta = U1652_TASKS[self.task_name]
            gt_path = dataset_root / meta["gt"]
            if not gt_path.is_file():
                raise FileNotFoundError(
                    f"Ground truth file not found: {gt_path}. Please run scripts/create_university1652_metadata.py first."
                )
            import numpy as np

            gt = np.load(gt_path, allow_pickle=True)
            self.ground_truth = [list(map(int, g)) for g in gt.tolist()]
        else:
            raise ValueError(f"Unknown u1652 task name inferred from feature file: {self.task_name}")

        self.topk_indices = self._compute_global_topk(self.features, k=max_k)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        indices = torch.cat([
            torch.tensor([idx], device=self.topk_indices.device),
            self.topk_indices[idx]
        ])
        return self.features[indices], idx

    def _compute_global_topk(self, feats: torch.Tensor, k: int) -> torch.Tensor:
        n = feats.shape[0]
        search_k = min(k + 1, n)

        if _FAISS_AVAILABLE:
            _, idx = faiss_knn_topk(feats, k=search_k, database=feats, metric="cosine", prefer_gpu=True)
            idx = idx.long()
        else:
            feats_norm = torch.nn.functional.normalize(feats, p=2, dim=1)
            sim = feats_norm @ feats_norm.t()
            _, idx = torch.topk(sim, k=search_k, dim=1, largest=True)

        return idx[:, 1:]


class UCADataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_cache_root: str = "cache/u1652",
        val_cache_root: str = "cache/u1652",
        train_batch_size: int = 64,
        val_batch_size: int = 32,
        num_workers: int = 4,
        max_k: int = 100,
        val_tasks: Sequence[str] = ("u1652_drone2satellite", "u1652_satellite2drone"),
    ):
        super().__init__()
        self.train_cache_root = Path(train_cache_root)
        self.val_cache_root = Path(val_cache_root)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.max_k = max_k
        self.val_tasks = list(val_tasks)

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: List[Dataset] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "train"):
            feature_names = self._discover_train_features()
            self.train_dataset = University1652UCADataset(
                dataset_path=str(self.train_cache_root),
                feature_names=feature_names,
                max_k=self.max_k,
            )

        if stage in (None, "fit", "validate"):
            val_files = self._discover_val_files()
            self.val_datasets = [
                U1652FeatureValDataset(f, max_k=self.max_k) for f in val_files
            ]

    def train_dataloader(self) -> DataLoader:
        if not self.train_dataset:
             raise RuntimeError("Train dataset not initialized.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            for ds in self.val_datasets
        ]

    def _discover_train_features(self) -> List[str]:
        d_path = self.train_cache_root / "drone"
        s_path = self.train_cache_root / "satellite"
        if not d_path.exists() or not s_path.exists():
            raise FileNotFoundError(f"Train directories not found in {self.train_cache_root}")

        def get_base(n: str) -> str:
            return re.sub(r'_\d+\.(h5|hdf5)$', '', n)

        d_files = {get_base(f.name) for f in d_path.glob("*.h5")}
        s_files = {get_base(f.name) for f in s_path.glob("*.h5")}

        return sorted(list(d_files & s_files))

    def _discover_val_files(self) -> List[Path]:
        files = []
        if not self.val_cache_root.exists():
             return []

        for f in self.val_cache_root.glob("*_features.h5"):
            if any(task in f.name for task in self.val_tasks):
                files.append(f)
        return sorted(files)
