from typing import Optional, Callable, Tuple, Any
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from src.utils.logger import print_rank_0


U1652_TASKS = {
	"u1652_drone2satellite": {
		"db": "u1652_drone2satellite_dbImages.npy",
		"q": "u1652_drone2satellite_qImages.npy",
		"gt": "u1652_drone2satellite_gt.npy",
	},
	"u1652_satellite2drone": {
		"db": "u1652_satellite2drone_dbImages.npy",
		"q": "u1652_satellite2drone_qImages.npy",
		"gt": "u1652_satellite2drone_gt.npy",
	},
	"u1652_drone2street": {
		"db": "u1652_drone2street_dbImages.npy",
		"q": "u1652_drone2street_qImages.npy",
		"gt": "u1652_drone2street_gt.npy",
	},
	"u1652_street2satellite": {
		"db": "u1652_street2satellite_dbImages.npy",
		"q": "u1652_street2satellite_qImages.npy",
		"gt": "u1652_street2satellite_gt.npy",
	},
}


class University1652Dataset(Dataset):
	"""University-1652 validation/test dataset for cross-view retrieval."""

	def __init__(
		self,
		dataset_path: Optional[str] = None,
		dataset_name: str = "drone2satellite",
		input_transform: Optional[Callable] = None,
	):
		self.input_transform = input_transform
		self.dataset_name = dataset_name

		if dataset_path is None:
			dataset_path = Path("data/university1652/test")
		else:
			dataset_path = Path(dataset_path)

		if dataset_name not in U1652_TASKS:
			raise ValueError(f"Unknown dataset_name '{dataset_name}'. Available: {list(U1652_TASKS.keys())}")

		if not dataset_path.is_dir():
			raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")

		meta = U1652_TASKS[dataset_name]
		db_path = dataset_path / meta["db"]
		q_path = dataset_path / meta["q"]
		gt_path = dataset_path / meta["gt"]

		if not db_path.is_file():
			raise FileNotFoundError(f"The file {db_path} does not exist. Please run scripts/create_university1652_metadata.py first.")
		if not q_path.is_file():
			raise FileNotFoundError(f"The file {q_path} does not exist.")
		if not gt_path.is_file():
			raise FileNotFoundError(f"The file {gt_path} does not exist.")

		self.dataset_path = dataset_path

		# Load image names and ground truth data
		self.dbImages = np.load(db_path)
		self.qImages = np.load(q_path)
		self.ground_truth = np.load(gt_path, allow_pickle=True)

		# Combine reference and query images (db first, then queries)
		self.image_paths = np.concatenate([self.dbImages, self.qImages])
		self.num_references = len(self.dbImages)
		self.num_queries = len(self.qImages)
		
		print_rank_0(f"University1652Dataset '{self.dataset_name}' initialized with {self.num_references} reference images and {self.num_queries} query images.")

	def __getitem__(self, index: int) -> Tuple[Any, int]:
		img_path = self.image_paths[index]
		full_path = self.dataset_path / img_path

		try:
			img = Image.open(full_path).convert("RGB")
		except Exception as e:
			print(f"Error loading image {full_path}: {e}")
			raise e

		if self.input_transform:
			img = self.input_transform(img)

		return img, index

	def __len__(self) -> int:
		return len(self.image_paths)