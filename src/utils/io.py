"""IO工具函数"""

import h5py
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.distributed as dist
import pytorch_lightning as pl

def load_config(yaml_path: str) -> Dict[str, Any]:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg

def load_npz(path: Path) -> Dict[str, Any]:
    """加载npz文件
    
    Args:
        path: npz文件路径
        
    Returns:
        data: 包含npz数据的字典
    """
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_features_h5(
    features: np.ndarray,
    image_names: List[str],
    save_path: Path,
    cache_dir: Path = None,
    **metadata
):
    """保存特征到h5文件
    
    Args:
        features: 特征数组 (N, D)
        image_names: 图像名称列表 (N,)
        save_path: 保存路径（相对于cache_dir）
        cache_dir: cache目录，默认为 'cache/'
        **metadata: 额外的元数据
    
    Examples:
        >>> feats = np.random.rand(100, 512)
        >>> names = [f'img_{i}.jpg' for i in range(100)]
        >>> save_features_h5(feats, names, 'gl3d_train.h5')
    """
    if cache_dir is None:
        cache_dir = Path("cache")
    
    # 创建cache目录
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 完整保存路径
    full_path = cache_dir / save_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为h5文件
    with h5py.File(full_path, "w") as f:
        # 保存特征
        f.create_dataset("features", data=features, compression="gzip")
        
        # 保存图像名称（字符串列表）
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("image_names", data=image_names, dtype=dt)
        
        # 保存元数据
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                f.attrs[k] = v
            elif isinstance(v, np.ndarray):
                f.create_dataset(f"metadata/{k}", data=v)
            else:
                # 其他类型转为字符串
                f.attrs[k] = str(v)


def load_features_h5(
    load_path: Path,
    cache_dir: Path = None
) -> Dict[str, Any]:
    """加载h5特征文件
    
    Args:
        load_path: 加载路径（相对于cache_dir）
        cache_dir: cache目录，默认为 'cache/'
        
    Returns:
        data: 包含features、image_names和metadata的字典
            - features: (N, D) array
            - image_names: (N,) list of str
            - metadata: dict
    
    Examples:
        >>> data = load_features_h5('gl3d_train.h5')
        >>> feats = data['features']
        >>> names = data['image_names']
    """
    if cache_dir is None:
        cache_dir = Path("cache")
    
    full_path = cache_dir / load_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Feature file not found: {full_path}")
    
    data = {}
    
    with h5py.File(full_path, "r") as f:
        # 加载特征
        data["features"] = f["features"][:]
        
        # 加载图像名称
        data["image_names"] = [name.decode("utf-8") if isinstance(name, bytes) else name 
                               for name in f["image_names"][:]]
        
        # 加载属性（元数据）
        data["metadata"] = dict(f.attrs)
        
        # 加载metadata数据集
        if "metadata" in f:
            for k in f["metadata"].keys():
                data["metadata"][k] = f[f"metadata/{k}"][:]
    
    return data


def load_vlad_init_h5(
    load_path: Path,
    cache_dir: Path = None,
) -> Dict[str, Any]:
    """Load VLAD init cache from H5.

    Required datasets:
        - centers: (K, C)
        - descriptors: (N, C)

    Args:
        load_path: Path to H5 (relative to cache_dir or absolute)
        cache_dir: Cache root for relative paths, default ``cache/``

    Returns:
        Dict with keys: ``centers``, ``descriptors``, ``metadata``
    """
    if cache_dir is None:
        cache_dir = Path("cache")

    load_path = Path(load_path)
    full_path = load_path if load_path.is_absolute() else Path(cache_dir) / load_path

    if not full_path.exists():
        raise FileNotFoundError(f"VLAD init cache file not found: {full_path}")

    data: Dict[str, Any] = {}
    with h5py.File(full_path, "r") as f:
        if "centers" not in f or "descriptors" not in f:
            raise RuntimeError(
                f"Invalid VLAD init H5: {full_path}. Required datasets: 'centers' and 'descriptors'."
            )
        data["centers"] = np.ascontiguousarray(f["centers"][:], dtype=np.float32)
        data["descriptors"] = np.ascontiguousarray(f["descriptors"][:], dtype=np.float32)
        data["metadata"] = dict(f.attrs)

    return data


def save_features_dict_h5(
    feature_dict: Dict[str, np.ndarray],
    save_path: Path,
    cache_dir: Path = None,
    **metadata
):
    """保存特征字典到h5文件
    
    Args:
        feature_dict: 特征字典 {image_name: feature}
        save_path: 保存路径（相对于cache_dir）
        cache_dir: cache目录，默认为 'cache/'
        **metadata: 额外的元数据
    
    Examples:
        >>> feat_dict = {'img1.jpg': np.random.rand(512), 
        ...              'img2.jpg': np.random.rand(512)}
        >>> save_features_dict_h5(feat_dict, 'features.h5')
    """
    # 转换为数组格式
    image_names = list(feature_dict.keys())
    features = np.stack([feature_dict[name] for name in image_names])
    
    save_features_h5(features, image_names, save_path, cache_dir, **metadata)


def load_features_dict_h5(
    load_path: Path,
    cache_dir: Path = None
) -> Dict[str, np.ndarray]:
    """加载h5特征文件为字典格式
    
    Args:
        load_path: 加载路径（相对于cache_dir）
        cache_dir: cache目录，默认为 'cache/'
        
    Returns:
        feature_dict: {image_name: feature}
    
    Examples:
        >>> feat_dict = load_features_dict_h5('features.h5')
        >>> feat = feat_dict['img1.jpg']
    """
    data = load_features_h5(load_path, cache_dir)
    
    # 转换为字典格式
    feature_dict = {}
    for name, feat in zip(data["image_names"], data["features"]):
        feature_dict[name] = feat
    
    return feature_dict


# 向后兼容的别名
def save_features(features: np.ndarray, path: Path, **metadata):
    """保存特征到npz文件（向后兼容）
    
    Args:
        features: 特征数组 (N, D)
        path: 保存路径
        **metadata: 额外的元数据
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features, **metadata)


def load_features(path: Path) -> Dict[str, Any]:
    """加载特征文件（向后兼容）
    
    Args:
        path: 特征文件路径
        
    Returns:
        data: 包含features和metadata的字典
    """
    return load_npz(path)


def gather_distributed_predictions(predictions: List[Dict[str, torch.Tensor]], trainer: pl.Trainer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather predictions from all GPU ranks and merge them.
    Assumes model.predict_step returns {'features': Tensor, 'index': Tensor}.
    """
    # 1. Local Concatenation
    local_feats = torch.cat([p['features'] for p in predictions], dim=0)
    local_idxs = torch.cat([p['indices'] for p in predictions], dim=0) # Ensure key matches predict_step

    # 2. If Single GPU/CPU, return directly
    if not (dist.is_available() and dist.is_initialized()):
        return local_feats, local_idxs

    # 3. DDP Handling (Move to CUDA for NCCL)
    device = torch.device(f"cuda:{trainer.local_rank}")
    local_feats = local_feats.to(device)
    local_idxs = local_idxs.to(device)

    # A. Get max size across all ranks (for padding)
    local_size = torch.tensor([local_feats.shape[0]], device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    max_size = max([s.item() for s in all_sizes])

    # B. Pad local tensors if necessary
    feat_dim = local_feats.shape[1]
    if local_size < max_size:
        pad_len = max_size - local_size
        local_feats = torch.cat([local_feats, torch.zeros(pad_len, feat_dim, device=device)], dim=0)
        local_idxs = torch.cat([local_idxs, torch.full((pad_len,), -1, device=device)], dim=0)

    # C. All Gather
    gathered_feats = [torch.zeros_like(local_feats) for _ in range(dist.get_world_size())]
    gathered_idxs = [torch.zeros_like(local_idxs) for _ in range(dist.get_world_size())]
    
    dist.all_gather(gathered_feats, local_feats)
    dist.all_gather(gathered_idxs, local_idxs)

    # 4. Merge & Clean (Move back to CPU)
    full_feats = torch.cat(gathered_feats, dim=0).cpu()
    full_idxs = torch.cat(gathered_idxs, dim=0).cpu()

    # Remove padding (indices == -1)
    valid_mask = full_idxs != -1
    return full_feats[valid_mask], full_idxs[valid_mask]