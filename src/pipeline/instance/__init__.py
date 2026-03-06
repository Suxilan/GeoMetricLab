"""Instance metric learning训练Pipeline模块"""

from .instance_datamodule import InstanceDataModule
from .instance_framework import InstanceFramework

__all__ = [
    "InstanceDataModule",
    "InstanceFramework",
]

