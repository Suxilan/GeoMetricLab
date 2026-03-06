
from typing import Optional, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset, Sampler
import numpy as np
from PIL import Image
from src.utils.logger import print_rank_0
GL3D_META = {
    "gl3d_test": {
        "db": "gl3d_test_dbImages.npy",
        "q": "gl3d_test_qImages.npy",
        "gt": "gl3d_test_gt.npy",
    }
    ,
    "gl3d_test_scene": {
        "db": "gl3d_test_dbImages.npy",
        "q": "gl3d_test_qImages.npy",
        "gt": "gl3d_test_gt.npy",
    }
}


class GL3DDataset(Dataset):
    """
    GL3D validation/test dataset.
    
    Args:
        dataset_path (str): Directory containing the dataset.
        input_transform (callable, optional): Optional transform to be applied on each image.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: str = "gl3d_test",
        input_transform: Optional[Callable] = None,
    ):
        self.input_transform = input_transform
        self.dataset_name = dataset_name

        if dataset_path is None:
            dataset_path = Path("data/GL3D/test")
        else:
            dataset_path = Path(dataset_path)
            
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")

        if dataset_name not in GL3D_META:
            raise ValueError(f"Unknown dataset_name '{dataset_name}'. Available: {list(GL3D_META.keys())}")

        meta = GL3D_META[dataset_name]

        # Check for metadata files
        db_path = dataset_path / meta["db"]
        q_path = dataset_path / meta["q"]
        gt_path = dataset_path / meta["gt"]
        
        if not db_path.is_file():
            raise FileNotFoundError(f"The file {db_path} does not exist. Please run scripts/create_gl3d_test_metadata.py first.")
        if not q_path.is_file():
            raise FileNotFoundError(f"The file {q_path} does not exist.")
        if not gt_path.is_file():
            raise FileNotFoundError(f"The file {gt_path} does not exist.")
        
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        
        # Load image names and ground truth data
        self.dbImages = np.load(db_path)
        self.qImages = np.load(q_path)
        self.ground_truth = np.load(gt_path, allow_pickle=True)

        # Combine reference and query images
        # Note: For GL3D, dbImages and qImages are typically identical (all images).
        # We concatenate them to follow the standard evaluation protocol where
        # features are extracted for both sets (even if identical).
        self.image_paths = self.dbImages
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

        print_rank_0(f"[{self.__class__.__name__}] initialized with {self.num_references} reference images and {self.num_queries} query images.")
    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, index) where image is a PIL image.
        """
        img_path = self.image_paths[index]
        full_path = self.dataset_path / img_path
        
        try:
            img = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Return a black image or raise error?
            # Raising error is better to catch issues
            raise e

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset.
        """
        return len(self.image_paths)


class GL3DSceneDataset(GL3DDataset):
    """GL3D dataset grouped by scene order (for scene-wise batching)."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: str = "gl3d_test_scene",
        input_transform: Optional[Callable] = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            input_transform=input_transform,
        )
        self.scene_groups = self._build_scene_groups(self.image_paths)

    @staticmethod
    def _build_scene_groups(image_paths: np.ndarray) -> list[list[int]]:
        groups: list[list[int]] = []
        if len(image_paths) == 0:
            return groups

        def scene_id(p: str) -> str:
            return str(p).split("/")[0]

        current_scene = scene_id(image_paths[0])
        current_group: list[int] = [0]
        for idx in range(1, len(image_paths)):
            s = scene_id(image_paths[idx])
            if s != current_scene:
                groups.append(current_group)
                current_group = [idx]
                current_scene = s
            else:
                current_group.append(idx)
        groups.append(current_group)
        return groups


class SceneBatchSampler(Sampler[list[int]]):
    """Yield batches within each scene; keep remainder batches if not divisible."""

    def __init__(
        self,
        dataset: GL3DSceneDataset,
        batch_size: int,
        drop_last: bool = False,
        shuffle_scenes: bool = False,
        shuffle_within_scene: bool = False,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle_scenes = bool(shuffle_scenes)
        self.shuffle_within_scene = bool(shuffle_within_scene)
        self.seed = int(seed)

    def _base_dataset(self):
        ds = self.dataset
        # unwrap DistributedSampler or other wrappers that expose `.dataset`
        while not hasattr(ds, "scene_groups") and hasattr(ds, "dataset"):
            ds = ds.dataset
        return ds

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        base = self._base_dataset()
        scene_groups = list(base.scene_groups)

        if self.shuffle_scenes:
            rng.shuffle(scene_groups)

        for group in scene_groups:
            indices = list(group)
            if self.shuffle_within_scene:
                rng.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        base = self._base_dataset()
        for group in base.scene_groups:
            if self.drop_last:
                total += len(group) // self.batch_size
            else:
                total += (len(group) + self.batch_size - 1) // self.batch_size
        return total
