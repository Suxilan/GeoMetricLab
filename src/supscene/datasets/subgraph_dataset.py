"""SubgraphDataset基类：用于子图数据集的抽象基类"""

import os
import math
import random
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from src.utils.logger import print_rank_0
from .subgraph_sampler import SubgraphSampler


class SubgraphDataset(Dataset, ABC):
    """子图数据集的抽象基类
    
    所有子图数据集应该继承此类，实现核心的子图采样和返回逻辑。
    子类只需要实现_prepare_dataset方法来加载场景图。
    """
    
    def __init__(
        self,
        n_sub: int = 256,
        sampler: Optional[SubgraphSampler] = None,
        transform: Optional[Any] = None,
        scenes_per_epoch: Optional[int] = None,
        samples_per_scene: Optional[int] = None,  
        adaptive_sampling: bool = True,
        min_images_per_scene: int = 50,
    ):
        """初始化SubgraphDataset
        
        Args:
            n_sub (int): 每个样本的子图节点数量
            sampler (Optional[SubgraphSampler]): 采样器，默认为uniform
            transform: 图像变换（可选）
            scenes_per_epoch (Optional[int]): 每个epoch的场景数量
            samples_per_scene (Optional[int]): 每个场景的固定样本数；如果为None且
            adaptive_sampling=True，则自适应计算每个场景的样本数
            adaptive_sampling (bool): 启用自适应每场景采样数量
            min_images_per_scene (int): 过滤掉图像数量少于阈值的场景
        """
        super().__init__()
        self.n_sub = n_sub
        self.transform = transform
        self.adaptive_sampling = adaptive_sampling
        self.min_images_per_scene = int(min_images_per_scene)
        
        # 子类需要设置这些属性
        self.scene_graphs: List[Any] = []
        self.scenes: List[Any] = []
        
        # 加载场景图（子类实现）
        self._prepare_dataset()
        
        # 过滤场景
        if self.min_images_per_scene > 0:
            self.scenes = [s for s in self.scene_graphs if s.N >= self.min_images_per_scene]
        else:
            self.scenes = self.scene_graphs
        
        if len(self.scenes) == 0:
            raise ValueError(
                f"No valid scenes found: min_images_per_scene={self.min_images_per_scene}. "
                f"Total scenes={len(self.scene_graphs)}. Lower the threshold or check dataset."
            )

        self.sampler = sampler or SubgraphSampler(mode="uniform")
        self.num_scenes = len(self.scenes)
        self.scenes_per_epoch = scenes_per_epoch if scenes_per_epoch is not None else self.num_scenes
        
        if samples_per_scene is None and adaptive_sampling:
            self.samples_per_scene_list = [self._adaptive_samples_per_scene(scene.N) for scene in self.scenes]
            self.samples_per_scene = 1  # unused in adaptive mode
        else:
            self.samples_per_scene = int(samples_per_scene or 1)
            self.samples_per_scene_list = None

        # 构建epoch的采样计划
        self._build_epoch_indices()
    
    def print(self, message: str):
        print_rank_0(f"[{self.__class__.__name__}] {message}")
    
    @abstractmethod
    def _prepare_dataset(self):
        """准备数据集：扫描目录并加载场景图
        
        子类必须实现此方法，将场景图加载到self.scene_graphs列表中。
        """
        pass
    
    def _adaptive_samples_per_scene(self, N_images: int, min_samples: int = 1, max_samples: int = 8) -> int:
        """启发式：更大的场景产生更多的子图样本
        
        Args:
            N_images (int): 场景中的图像数量 (N)
            min_samples (int): 下界
            max_samples (int): 上界
        
        Returns:
            int: 该场景的样本数量
        """
        N = int(N_images)
        if N <= self.n_sub:
            return int(min_samples)
        cov = N / float(self.n_sub)
        s = int(np.ceil(np.sqrt(cov) * 1.5))  # sqrt dampens growth
        return int(np.clip(s, min_samples, max_samples))

    def _build_epoch_indices(self) -> None:
        """构建该epoch的场景索引列表"""
        if self.samples_per_scene_list is not None:
            pool = random.sample(range(self.num_scenes), k=min(self.scenes_per_epoch, self.num_scenes))
            scene_indices: List[int] = []
            for i in pool:
                k = self.samples_per_scene_list[i]
                scene_indices.extend([i] * k)
            random.shuffle(scene_indices)
            self.epoch_scene_indices = scene_indices
            self.total_samples = len(scene_indices)
        else:
            self.total_samples = int(self.scenes_per_epoch) * int(self.samples_per_scene)
            reps = math.ceil(self.total_samples / max(1, self.num_scenes))
            scene_indices = (list(range(self.num_scenes)) * reps)[: self.total_samples]
            random.shuffle(scene_indices)
            self.epoch_scene_indices = scene_indices

    def reshuffle_epoch(
        self,
        epoch: Optional[int] = None,
        scenes_per_epoch: Optional[int] = None,
        samples_per_scene: Optional[int] = None,
    ) -> None:
        """在epoch边界重建采样计划
        
        Args:
            epoch (Optional[int]): 忽略；用于外部调度器的钩子
            scenes_per_epoch (Optional[int]): 覆盖每个epoch的场景数
            samples_per_scene (Optional[int]): 覆盖每个场景的样本数；禁用自适应模式
        """
        if scenes_per_epoch is not None:
            self.scenes_per_epoch = int(scenes_per_epoch)
        if samples_per_scene is not None:
            self.samples_per_scene = int(samples_per_scene)
            self.samples_per_scene_list = None  # disable adaptive when manually set
        self._build_epoch_indices()
    
    def _get_scene_graph(self, scene_idx: int):
        """获取场景图对象"""
        return self.scenes[scene_idx]
    
    def _sample_subgraph(self, idx: int):
        """从场景中采样子图"""
        scene_idx = self.epoch_scene_indices[idx]
        G = self.scenes[scene_idx]
        node_idx = self.sampler.sample(G, self.n_sub)  # (n,)
        return node_idx, G
    
    def _load_image(self, path: str) -> torch.Tensor:
        """加载RGB图像并应用变换
        
        Args:
            path (str): 图像文件路径
        
        Returns:
            torch.Tensor: (3, H, W) float tensor
        """
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else torch.from_numpy(
            np.array(img)
        ).permute(2, 0, 1).float() / 255.0
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个样本，包含子图和可选的图像
        
        Args:
            idx (int): 样本索引
        
        Returns:
            Dict: {
                'scene_id': str,
                'node_idx': LongTensor (n,),
                'overlap': FloatTensor (n, n),
                'image_paths': List[str],
                'images': FloatTensor (n, 3, H, W) or None,
                'num_nodes': int,
            }
        """
        node_idx, scene_graph = self._sample_subgraph(idx)
        n = int(node_idx.shape[0])
        
        # 获取密集重叠矩阵
        O_np = scene_graph.dense_overlap(node_idx, add_self=True)  # (n, n)
        
        # 获取图像路径
        paths = [scene_graph.image_paths[i] for i in node_idx.tolist()]
        
        # 加载图像（如果提供了transform）
        imgs = None
        if self.transform is not None:
            imgs = torch.stack([self._load_image(p) for p in paths], dim=0)  # (n, 3, H, W)
        
        return {
            "scene_id": os.path.basename(scene_graph.scene_dir),
            "node_idx": torch.from_numpy(node_idx).long(),
            "overlap": torch.from_numpy(O_np),
            "image_paths": paths,
            "images": imgs,
            "num_nodes": n,
        }

    def __len__(self):
        return self.total_samples
