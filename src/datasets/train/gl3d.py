
import random
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.supscene.datasets.scenegraph import SceneGraph
from src.supscene.datasets.subgraph_dataset import SubgraphDataset
from src.supscene.datasets.subgraph_sampler import SubgraphSampler
from src.utils.logger import print_rank_0

class GL3DSubgraphDataset(SubgraphDataset):
    """GL3D子图数据集：每个样本返回一个子图（可能来自不同场景）
    
    特性:
        - 可选的图像加载（用于微调编码器）
        - 每个epoch重新打乱/重建采样计划
        - 使用与GL3DTrainDataset相同的场景扫描方式
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: str = "gl3d_subgraph",
        input_transform: Optional[Any] = None,
        n_sub: int = 256,
        iou_th: float = 0.2,
        scenes_per_epoch: Optional[int] = None,
        samples_per_scene: Optional[int] = None,  
        adaptive_sampling: bool = True,
        min_images_per_scene: int = 50,
    ):
        """数据集构造函数
        
        Args:
            root (Union[str, Path]): 数据集根目录
            n_sub (int): 每个样本的子图节点数量
            ov_thresh (Dict[str, Dict[str, Any]]): 重叠阈值配置，用于初始化sampler
            transform: 图像变换（可选）
            scenes_per_epoch (Optional[int]): 每个epoch的场景数量
            samples_per_scene (Optional[int]): 每个场景的固定样本数；如果为None且
                adaptive_sampling=True，则自适应计算每个场景的样本数
            adaptive_sampling (bool): 启用自适应每场景采样数量
            min_images_per_scene (int): 过滤掉图像数量少于阈值的场景
        """
        if dataset_path is None:
            print_rank_0("Using default GL3D dataset path: data/GL3D/train")
            dataset_path = Path("data/GL3D/train")
        else:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")
            
        sampler = SubgraphSampler(mode="anchor_expand", iou_th=iou_th)

        # Ensure dataset_path is available for parent initializer which may
        # call _prepare_dataset() and expect self.dataset_path to exist.
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        super().__init__(
            n_sub=n_sub,
            sampler=sampler,
            transform=input_transform,
            scenes_per_epoch=scenes_per_epoch,
            samples_per_scene=samples_per_scene,
            adaptive_sampling=adaptive_sampling,
            min_images_per_scene=min_images_per_scene,
        )
    
    def _prepare_dataset(self):
        """准备数据集：扫描目录并加载场景图（参考GL3DTrainDataset）"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.dataset_path}")

        # List all subdirectories (scenes) - 与GL3DTrainDataset相同的方式
        scene_dirs = sorted([
            d for d in self.dataset_path.iterdir() if d.is_dir()
        ])

        self.print(f"Found {len(scene_dirs)} scenes in {self.dataset_path}. Loading graphs...")

        # Load SceneGraphs
        self.scene_graphs: List[SceneGraph] = []
        for s_dir in tqdm(scene_dirs, desc="Loading SceneGraphs", leave=False):
            try:
                sg = SceneGraph(str(s_dir))
                
                if sg.N == 0:
                    continue
                    
                self.scene_graphs.append(sg)
                    
            except Exception as e:
                self.print(f"Skipping scene {s_dir.name}: {e}")

        self.print(f"Loaded {len(self.scene_graphs)} scenes")

class GL3DInstanceDataset(Dataset):
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: str = "gl3d_instance",
        input_transform: Optional[Any] = None,
        pos_th: float = 0.3,
        pos_nums: Optional[int] = None,
    ):
        if dataset_path is None:
            print_rank_0("Using default GL3D dataset path: data/GL3D/train")
            dataset_path = Path("data/GL3D/train")
        else:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"The directory {dataset_path} does not exist. Please check the path.")

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.transform = input_transform
        self.pos_th = pos_th
        self.pos_nums = pos_nums

        self.scene_graphs: List[SceneGraph] = []
        self.pos_lists: List[List[List[int]]] = []
        self.anchor_lists: List[List[int]] = []

        self._prepare_dataset()
           
        self.num_scenes = len(self.scene_graphs)
        self.num_images = sum(sg.N for sg in self.scene_graphs)

    def _prepare_dataset(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.dataset_path}")

        scene_dirs = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])

        for s_dir in tqdm(scene_dirs, desc="Loading SceneGraphs", leave=False):
            try:
                sg = SceneGraph(str(s_dir))
                if sg.N < 3:
                    continue

                pos_adj = sg.neighbor_lists(self.pos_th)
                for a in range(sg.N):
                    if len(pos_adj[a]) > 1:
                        pos_adj[a] = sorted(set(pos_adj[a]))

                anchors: List[int] = []
                for a in range(sg.N):
                    pos = pos_adj[a]
                    if len(pos) == 0:
                        continue
                    if (sg.N - 1 - len(pos)) > 0:
                        anchors.append(a)

                if len(anchors) == 0:
                    continue

                self.scene_graphs.append(sg)
                self.pos_lists.append(pos_adj)
                self.anchor_lists.append(anchors)
            except Exception:
                continue

        if len(self.scene_graphs) == 0:
            raise ValueError(f"No valid scenes found under {self.dataset_path} with pos_th={self.pos_th}")

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def __len__(self) -> int:
        # 关键：长度就是场景数
        return len(self.scene_graphs)

    def __getitem__(self, scene_idx: int):
        sg = self.scene_graphs[scene_idx]
        
        # 1) 选 Anchor
        anchor_idx = random.choice(self.anchor_lists[scene_idx])
        
        # 2) 选 Positive
        pos_candidates = self.pos_lists[scene_idx][anchor_idx]
        selected_pos = random.choices(pos_candidates, k=self.pos_nums)
        pos_set = set(pos_candidates)

        # 3) 选严格 Negative (快速采样)
        for _ in range(32):
            neg_idx = random.randrange(sg.N)
            if neg_idx != anchor_idx and neg_idx not in pos_set:
                break
        else:
            # 极少触发的兜底方案
            neg_pool = [j for j in range(sg.N) if j != anchor_idx and j not in pos_set]
            neg_idx = random.choice(neg_pool)

        # 4) 加载图像
        all_indices = [anchor_idx] + selected_pos + [neg_idx]
        imgs = torch.stack([self._load_image(sg.image_paths[i]) for i in all_indices], dim=0)

        return {"images": imgs}

def gl3d_instance_collate(batch: List[dict]):
    """Collate triples into (B*M, C, H, W) with per-item labels.

    Anchor/positive share the batch index label; negatives are unique via 1000 + batch_idx.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch passed to gl3d_instance_collate")

    images = torch.cat([item["images"] for item in batch], dim=0)
    B = len(batch)
    M = batch[0]["images"].size(0)

    labels = torch.empty(B * M, dtype=torch.long)
    for b in range(B):
        offset = b * M
        labels[offset : offset + M - 1] = b
        labels[offset + M - 1] = 1000 + b
        
    return images, labels