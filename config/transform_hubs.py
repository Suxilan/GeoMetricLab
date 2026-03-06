import torch
from torchvision.transforms import v2 as T2
"""Centralized dataset transforms.

Provides ready-to-use train/val transforms per dataset key. Each mapping
value is a callable `fn(img_size) -> torchvision.transforms.Compose`.

Usage:
    from src.datasets.transform import get_transform
    t = get_transform("gl3d", img_size=512, train=False)
"""


def _base_val_transform(img_size: tuple[int, int]):
    return T2.Compose([
        T2.ToImage(),
        T2.Resize(size=img_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def gl3d_transform(img_size: tuple[int, int]):
    # Reasonable default for retrieval / metric-learning training
    return T2.Compose([
        T2.ToImage(),
        T2.Resize(size=img_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
        # T2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), interpolation=T2.InterpolationMode.BICUBIC),
        # T2.RandomHorizontalFlip(p=0.5),
        # T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def u1652_transform(img_size: tuple[int, int]):
    """
    针对 University-1652 的差异化数据增强
    - Satellite: 保持朝向稳定，侧重颜色色彩抖动。
    - Drone: 加入随机旋转 (0-360)，模拟无人机绕飞。
    - Street/Google: 侧重水平翻转。
    """
    # 基础公共部分
    base_ops = [
        T2.ToImage(),
        # T2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), interpolation=T2.InterpolationMode.BICUBIC),
        # T2.RandomHorizontalFlip(p=0.5),
        T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # 针对 Drone 的特殊处理：增加 360 度随机旋转
    drone_ops = [
        T2.ToImage(),
        # T2.RandomRotation(180), # 模拟任意角度绕飞
        # T2.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), interpolation=T2.InterpolationMode.BICUBIC),
        # T2.RandomHorizontalFlip(p=0.5),
        # T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return {
        "satellite": T2.Compose(base_ops),
        "drone": T2.Compose(drone_ops),
        "street": T2.Compose(base_ops),
        "google": T2.Compose(base_ops)
    }

TRAIN_TRANSFORMS = {
    "gl3d": gl3d_transform,
    "u1652": u1652_transform
}


VAL_TRANSFORMS = {
    "gl3d": _base_val_transform,
    "u1652": _base_val_transform,
}


def get_transform(dataset_key: str, img_size: tuple[int, int], train: bool = False):
    """Return a torchvision transform Compose for given dataset key and image size.

    - `dataset_key` selects the predefined config (falls back to 'default').
    - `img_size` is the target size (int).
    - `train` chooses train or val transforms.
    """
    key = dataset_key if dataset_key in (TRAIN_TRANSFORMS if train else VAL_TRANSFORMS) else None
    if key is None:
        assert False, f"Unknown dataset_key '{dataset_key}' for {'train' if train else 'val'} transforms. \n\
        Available keys: {list(TRAIN_TRANSFORMS.keys()) if train else list(VAL_TRANSFORMS.keys())}"
    if train:
        return TRAIN_TRANSFORMS[key](img_size)
    else:
        return VAL_TRANSFORMS[key](img_size)

__all__ = ["get_transform", "TRAIN_TRANSFORMS", "VAL_TRANSFORMS"]