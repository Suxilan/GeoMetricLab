"""SupScene工具模块：提供overlap-aware的度量学习组件。

主要组件：
- datasets: SubgraphDataset, SubgraphSampler, make_pad_collate
- losses: SupConLoss

注意：GL3DSubgraphDataset 已迁移到 GeoMetricLab/datasets/GL3D/dataset.py
"""

from .datasets import (
    SubgraphDataset,
    SubgraphSampler,
    make_pad_collate,
    SceneGraph,
)
from .losses import SupConLoss

__all__ = [
    # Datasets
    "SubgraphDataset",
    "SubgraphSampler",
    "make_pad_collate",
    "SceneGraph",
    # Losses
    "SupConLoss",
]

