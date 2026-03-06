import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, List, Tuple
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as F
from src.utils.io import load_features_h5

class University1652InstanceDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = "u1652_instance",
        views: Optional[Sequence[str]] = None,
        input_transform: Optional[Dict] = None,
        n_samples: int = 1  # 除了satellite外，其他视图采样的张数
    ):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.views = views if views else ["satellite", "drone"]
        self.input_transform = input_transform if input_transform else {}
        self.n_samples = n_samples

        # 1. 自动扫描类别 (以 views 中的第一个视角作为基准进行 ID 发现)
        anchor_view = self.views[0]
        self.classes = sorted(os.listdir(os.path.join(dataset_path, anchor_view)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # 2. 预构建路径索引：{view: {class_id: [path1, path2, ...]}}
        self.samples = {view: {} for view in self.views}
        self._total_images = 0
        
        for view in self.views:
            view_dir = os.path.join(dataset_path, view)
            for cls in self.classes:
                cls_dir = os.path.join(view_dir, cls)
                if os.path.isdir(cls_dir):
                    imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    self.samples[view][cls] = imgs
                    self._total_images += len(imgs)

        # 按照要求固定的属性
        self.num_classes = len(self.classes)
        self.num_scenes = self.num_classes
        self.num_images = self._total_images

    def _get_single_view_data(self, view_name: str, cls_name: str):
        """处理具体视角的采样与增强逻辑"""
        all_paths = self.samples[view_name].get(cls_name, [])
        if not all_paths:
            return None

        # 采样策略：satellite 固定 1 张，其余视角采样 n_samples 张
        count = 1 if view_name == "satellite" else self.n_samples
        
        # 核心逻辑：支持重复采样 (replacement=True)
        selected_paths = random.choices(all_paths, k=count)
        
        imgs = []
        transform = self.input_transform.get(view_name)
        
        for p in selected_paths:
            img = Image.open(p).convert("RGB")
            if transform:
                img = transform(img)
            imgs.append(img)
            
        return imgs[0] if count == 1 else imgs

    def __len__(self) -> int:
        return self.num_classes

    def __getitem__(self, index: int) -> Dict[str, Any]:
        cls_name = self.classes[index]
        label = self.class_to_idx[cls_name]
        
        out = {"label": label, "id": cls_name}
        for view in self.views:
            out[view] = self._get_single_view_data(view, cls_name)
            
        return out


def u1652_instance_collate(batch: list) -> tuple:
    """Collate function for University1652 training dataset.

    - Converts per-sample per-view images into a single tensor of shape
      (B * N, C, H, W) where N is number of images per sample (assumed
      constant across the batch).
    - Returns labels tensor of shape (B,).

    The dataset may return PIL.Image.Image objects or already-transformed
    torch.Tensor instances; this collate will attempt to convert PIL images
    to float tensors in [0,1].
    """
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    all_images = []
    all_labels = []

    exclude_keys = {"label", "id", "paths", "dataset", "views"}
    view_keys = [k for k in batch[0].keys() if k not in exclude_keys]

    for sample in batch:
        label = sample["label"]
        sample_imgs = []

        for v in view_keys:
            val = sample.get(v)
            if val is None: continue
            
            if isinstance(val, list):
                sample_imgs.extend(val)
            else:
                sample_imgs.append(val)

        all_images.extend(sample_imgs)
        all_labels.extend([label] * len(sample_imgs))

    images = torch.stack(all_images, dim=0) 
    labels = torch.as_tensor(all_labels, dtype=torch.long)

    return images, labels


try:
    from src.rerank.faiss_utils import faiss_knn_topk
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass(frozen=True)
class _UCACache:
    """In-memory cache for one feature set."""
    feature_name: str
    drone_feats: torch.Tensor      # (Nd, 512)
    sat_feats: torch.Tensor        # (Ns, 512)
    global_feats: torch.Tensor     # (Nd+Ns, 512)
    # 预计算的 Cross-View 检索结果
    d2s_topk: torch.Tensor         # (Nd, K)
    s2d_topk: torch.Tensor         # (Ns, K)
    d2s_labels: torch.Tensor       # (Nd, K) bool, True 表示同 ID
    s2d_labels: torch.Tensor       # (Ns, K) bool, True 表示同 ID


class University1652UCADataset(Dataset):
    """UCA training dataset for University1652 (feature-based).

    - Loads paired drone/satellite H5 features.
    - Samples one feature set per iteration as augmentation.
    """

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str = "u1652_uca",
        feature_names: Optional[Sequence[str]] = None,
        max_k: int = 100,
        verbose: bool = True
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.dataset_name = str(dataset_name)
        self.drone_dir = self.dataset_path / "drone"
        self.sat_dir = self.dataset_path / "satellite"
        
        if not feature_names:
            raise ValueError("feature_names 不能为空")
        self.feature_names = list(feature_names)
        self.max_k = max_k
        self.verbose = verbose

        # Validate dirs
        if not self.drone_dir.is_dir() or not self.sat_dir.is_dir():
            raise FileNotFoundError(f"特征目录不存在: {self.drone_dir} 或 {self.sat_dir}")

        # All features are kept in memory for speed.
        self._caches: Dict[str, _UCACache] = {}
        
        # pair_id -> indices in each view
        self._paired_ids: List[str] = []
        self._drone_id_to_indices: Dict[str, List[int]] = {}
        self._sat_id_to_indices: Dict[str, List[int]] = {}
        self._drone_ids: List[str] = []
        self._sat_ids: List[str] = []
        self._drone_valid_indices: List[int] = []

        # For metric learning / PML
        self.class_to_idx: Dict[str, int] = {}

        # Use the first feature set as the reference ordering.
        base_drone_ids: Optional[List[str]] = None
        base_sat_ids: Optional[List[str]] = None

        for fname in self.feature_names:
            if self.verbose:
                print(f"[UCA Dataset] Loading feature: {fname} ...")
                
            cache, d_ids, s_ids = self._build_cache_for_feature(fname)
            self._caches[fname] = cache

            # Build id maps using the first feature set.
            if base_drone_ids is None:
                base_drone_ids = d_ids
                base_sat_ids = s_ids
                self._drone_ids = list(d_ids)
                self._sat_ids = list(s_ids)
                
                self._drone_id_to_indices = self._build_id_map(d_ids)
                self._sat_id_to_indices = self._build_id_map(s_ids)
                
                common_ids = set(self._drone_id_to_indices.keys()) & set(self._sat_id_to_indices.keys())
                self._paired_ids = sorted(list(common_ids))
                
                if not self._paired_ids:
                    raise RuntimeError("未找到 Drone 和 Satellite 共有的 ID，无法配对训练！")
                
                if self.verbose:
                    print(f"[UCA Dataset] Found {len(self._paired_ids)} paired locations.")

                self.class_to_idx = {sid: i for i, sid in enumerate(self._paired_ids)}
                # Train indexing: iterate over drone samples; for each drone, sample a satellite from the same scene.
                self._drone_valid_indices = [
                    i for i, sid in enumerate(self._drone_ids) if sid in self._sat_id_to_indices
                ]
                if not self._drone_valid_indices:
                    raise RuntimeError("未找到可配对的 drone 样本（无法从同 scene 采样 satellite）。")
            else:
                # Ensure consistent ordering across different feature files.
                if d_ids != base_drone_ids or s_ids != base_sat_ids:
                    raise RuntimeError(
                        f"特征集 {fname} 的 ID 顺序与基准特征集不一致！"
                        "请确保所有 H5 文件是按相同顺序生成的。"
                    )

    def __len__(self) -> int:
        # Iterate over drone samples (not scene pairs) for higher throughput.
        return len(self._drone_valid_indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Feature-level augmentation: sample one feature set each iteration.
        feat_name = random.choice(self.feature_names)
        cache = self._caches[feat_name]

        # Index over drone samples; then sample a satellite from the same scene.
        if index < 0 or index >= len(self._drone_valid_indices):
            raise IndexError(f"Index {index} out of range for dataset of length {len(self)}")
        drone_idx = int(self._drone_valid_indices[index])
        pair_id = self._drone_ids[drone_idx]
        sat_idx = random.choice(self._sat_id_to_indices[pair_id])

        d_query = cache.drone_feats[drone_idx]
        d2s_idx = cache.d2s_topk[drone_idx]
        d2s_neigh = cache.global_feats[d2s_idx]

        seq_drone = d_query.new_zeros((int(self.max_k) + 1, d_query.shape[-1]))
        seq_drone[0] = d_query
        seq_drone[1:] = d2s_neigh

        is_pos_d2s = cache.d2s_labels[drone_idx]

        s_query = cache.sat_feats[sat_idx]
        s2d_idx = cache.s2d_topk[sat_idx]
        s2d_neigh = cache.global_feats[s2d_idx]

        seq_sat = s_query.new_zeros((int(self.max_k) + 1, s_query.shape[-1]))
        seq_sat[0] = s_query
        seq_sat[1:] = s2d_neigh

        is_pos_s2d = cache.s2d_labels[sat_idx]

        return {
            "drone_seq": seq_drone,          # (K+1, 512)
            "drone_labels": is_pos_d2s,      # (K,)
            "sat_seq": seq_sat,              # (K+1, 512)
            "sat_labels": is_pos_s2d,        # (K,)
            "scene_id": pair_id,             # str
            "label": torch.tensor(self.class_to_idx[pair_id], dtype=torch.long),
        }

    # ================= 内部构建逻辑 =================

    def _build_cache_for_feature(self, fname: str) -> Tuple[_UCACache, List[str], List[str]]:
        """Load one feature set and precompute top-K + labels."""
        d_path, s_path = self._resolve_paths(fname)

        # 加载 H5 (假设 key 为 'features' 和 'image_names')
        d_feats, d_names = self._load_h5(d_path)
        s_feats, s_names = self._load_h5(s_path)

        # 解析 ID
        d_ids = [self._infer_id(n) for n in d_names]
        s_ids = [self._infer_id(n) for n in s_names]

        # 全局库：drone + satellite
        global_feats = torch.cat([d_feats, s_feats], dim=0)
        global_ids = d_ids + s_ids
        d_global_index = torch.arange(d_feats.shape[0], dtype=torch.long)
        s_global_index = torch.arange(s_feats.shape[0], dtype=torch.long) + int(d_feats.shape[0])

        # 预计算 TopK 和 Labels (核心耗时步)
        # 在全局库上检索，注意排除自身
        d2s_topk, d2s_labels = self._compute_topk(
            query_feats=d_feats,
            query_ids=d_ids,
            query_global_index=d_global_index,
            global_feats=global_feats,
            global_ids=global_ids,
        )

        s2d_topk, s2d_labels = self._compute_topk(
            query_feats=s_feats,
            query_ids=s_ids,
            query_global_index=s_global_index,
            global_feats=global_feats,
            global_ids=global_ids,
        )

        cache = _UCACache(
            feature_name=fname,
            drone_feats=d_feats,
            sat_feats=s_feats,
            global_feats=global_feats,
            d2s_topk=d2s_topk,
            s2d_topk=s2d_topk,
            d2s_labels=d2s_labels,
            s2d_labels=s2d_labels
        )
        return cache, d_ids, s_ids

    def _compute_topk(
        self,
        query_feats: torch.Tensor,
        query_ids: List[str],
        query_global_index: torch.Tensor,
        global_feats: torch.Tensor,
        global_ids: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute global top-K (exclude self) and boolean labels (same scene ID => positive)."""
        if global_feats.shape[0] <= self.max_k:
            raise ValueError(
                f"global size ({global_feats.shape[0]}) <= max_k ({self.max_k}); cannot build fixed-K sequences"
            )

        k = int(self.max_k)
        query_global_index = query_global_index.to(dtype=torch.long)

        # 1) 全局检索（先取 k+1，之后去掉自身）
        search_k = min(k + 1, int(global_feats.shape[0]))
        if FAISS_AVAILABLE:
            _, idx = faiss_knn_topk(
                query_feats,
                k=search_k,
                database=global_feats,
                metric="cosine",
                prefer_gpu=True,
            )
            idx = idx.long()
        else:
            q = F.normalize(query_feats, p=2, dim=1)
            g = F.normalize(global_feats, p=2, dim=1)
            sim = torch.mm(q, g.t())
            _, idx = torch.topk(sim, k=search_k, dim=1, largest=True)

        # 2) 排除自身索引，并截断到固定 K
        topk_rows: List[torch.Tensor] = []
        for row, self_i in zip(idx, query_global_index):
            row_wo_self = row[row != self_i]
            topk_rows.append(row_wo_self[:k])
        topk_idx = torch.stack(topk_rows, dim=0)  # (Nq,K)

        # 3) Labels：同 scene id 记为 True
        unique_ids = sorted(set(global_ids))
        id_to_code = {uid: i for i, uid in enumerate(unique_ids)}
        q_codes = torch.tensor([id_to_code[uid] for uid in query_ids], dtype=torch.long)
        g_codes = torch.tensor([id_to_code[uid] for uid in global_ids], dtype=torch.long)
        labels = q_codes.unsqueeze(1) == g_codes[topk_idx]
        
        return topk_idx, labels

    def _load_h5(self, path: Path) -> Tuple[torch.Tensor, List[str]]:
        """Load H5 via project IO helpers."""
        if not path.is_file():
            raise FileNotFoundError(f"文件未找到: {path}")

        data = load_features_h5(path, cache_dir=Path("/"))
        feats = torch.from_numpy(np.asarray(data["features"], dtype=np.float32))
        names = [str(n) for n in data.get("image_names", [])]

        if feats.ndim != 2 or feats.shape[1] != 512:
            raise ValueError(f"特征维度异常: {tuple(feats.shape)}，期望 (N, 512)。")
        if len(names) != feats.shape[0]:
            raise ValueError("image_names 数量与 features 行数不一致")

        return feats, names

    def _infer_id(self, name: str) -> str:
        """Parse scene id from image name/path (e.g. '0701_xxx' -> '0701')."""
        # 1. 尝试从路径父目录获取 (如果 name 是路径)
        if '/' in name:
            parent = name.split('/')[-2]
            # 简单的 heuristic: ID 通常是数字
            if parent.isdigit():
                return parent
        
        # 2. 从文件名获取
        stem = name.split('/')[-1] # 获取文件名
        # 去掉扩展名
        if '.' in stem:
            stem = stem.rsplit('.', 1)[0]
            
        # 分割符策略
        for sep in ['_', '-']:
            if sep in stem:
                return stem.split(sep)[0]
        
        return stem

    def _resolve_paths(self, feat_name: str) -> Tuple[Path, Path]:
        """根据简名 (e.g. 'dinov2') 自动寻找 drone 和 satellite 下的文件"""
        # 允许输入 'features_dinov2.h5' 或 'dinov2'
        
        def find_file(directory: Path) -> Path:
            # 精确匹配
            p = directory / feat_name
            if p.is_file(): return p
            p = directory / f"{feat_name}.h5"
            if p.is_file(): return p
            
            # 模糊匹配 (features_*{feat_name}*.h5)
            candidates = list(directory.glob(f"*{feat_name}*.h5"))
            if len(candidates) > 0:
                # 优先选最短的（通常最匹配），或者按字母序
                return sorted(candidates, key=lambda x: len(x.name))[0]
            
            raise FileNotFoundError(f"在 {directory} 中未找到匹配 {feat_name} 的文件")

        return find_file(self.drone_dir), find_file(self.sat_dir)

    def _build_id_map(self, ids: List[str]) -> Dict[str, List[int]]:
        """构建 ID -> Indices 的倒排索引"""
        mapping = {}
        for idx, uid in enumerate(ids):
            if uid not in mapping:
                mapping[uid] = []
            mapping[uid].append(idx)
        return mapping