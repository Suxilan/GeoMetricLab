"""SceneGraph：场景图加载和处理

本模块要求场景数据遵循以下目录结构和文件格式：

场景目录结构 (scene_dir/):
    scene_dir/
    ├── gt.npz                    # 必需：包含稀疏重叠矩阵的COO格式数据
    ├── image_index.json          # 可选：图像名称列表（优先使用）
    └── images/                   # 可选：图像文件目录（作为回退）
        ├── image1.jpg
        ├── image2.jpg
        └── ...

gt.npz 文件格式要求：
    必须包含以下字段（支持多种命名方式）：
    
    1. 边的源节点索引（二选一）：
       - "row" 或 "rows": np.ndarray, dtype=int64, shape=(E,)
         E为边的数量，值为节点索引（0到N-1）
    
    2. 边的目标节点索引（二选一）：
       - "col" 或 "cols": np.ndarray, dtype=int64, shape=(E,)
         与row/rows对应，值为节点索引（0到N-1）
    
    3. 边的权重/重叠值（三选一）：
       - "weight" 或 "val" 或 "vals": np.ndarray, dtype=float32, shape=(E,)
         重叠权重值，范围应在[0, 1]之间
    
    4. 图像名称列表（可选，用于确定图像顺序）：
       - "images" 或 "filenames": np.ndarray, dtype=object, shape=(N,)
         字符串数组，包含所有图像文件名（不含路径）
         如果提供，必须与节点数量N一致

image_index.json 文件格式（可选，优先使用）：
    支持三种格式：
    
    格式1 - 列表格式：
        ["image1.jpg", "image2.jpg", "image3.jpg", ...]
        直接按顺序列出所有图像文件名
    
    格式2 - id_to_name映射：
        {
            "id_to_name": ["image1.jpg", "image2.jpg", "image3.jpg", ...]
        }
        按id顺序排列的图像名称列表
    
    格式3 - name_to_id映射：
        {
            "name_to_id": {
                "image1.jpg": 0,
                "image2.jpg": 1,
                "image3.jpg": 2,
                ...
            }
        }
        图像名称到id的映射，会按id排序

图像顺序确定优先级：
    1. image_index.json（如果存在）
    2. gt.npz中的"images"或"filenames"字段（如果存在）
    3. images/目录下的文件列表（按文件名排序）

注意事项：
    - gt.npz中的row/col索引必须从0开始，连续编号到N-1
    - 重叠矩阵会自动对称化（添加反向边）
    - 自环（row[i] == col[i]）会被自动移除
    - 图像文件路径为 scene_dir/images/{image_name}
"""

import json
import os
from typing import List, Tuple
import numpy as np


class SceneGraph:
    """Load per-scene graph: image list + sparse overlap matrix (COO).

    Notes:
        - Supports gt.npz/overlap.npz using rows/cols/vals or legacy row/col/val names.
        - Reads the image order from images.txt, the npz metadata, or image_index.json.
    """

    def __init__(self, scene_dir: str):
        self.scene_dir = scene_dir
        self.images_dir = os.path.join(scene_dir, "images")
        self.image_index_file = os.path.join(scene_dir, "image_index.json")
        self.gt_npz = self._find_gt()
        
        # Load Metadata and Overlap Matrix
        with np.load(self.gt_npz, allow_pickle=True) as data:
            # 1. Load Image Names
            self.image_names = self._load_image_names(data)
            self.image_paths = [os.path.join(self.images_dir, name) for name in self.image_names]
            self.N = len(self.image_names)

            # 2. Load Sparse Matrix (COO)
            row = self._read_npz_field(data, ["row", "rows"], np.int64)
            col = self._read_npz_field(data, ["col", "cols"], np.int64)
            val = self._read_npz_field(data, ["weight", "val", "vals"], np.float32)

        # 3. Symmetrize and Clean Matrix
        # Concatenate (row, col) with (col, row) to ensure symmetry
        rows = np.concatenate([row, col])
        cols = np.concatenate([col, row])
        vals = np.concatenate([val, val])
        
        # Remove self-loops and duplicates (keeping max overlap) if necessary, 
        # but for simple symmetric adjacency, masking is usually sufficient.
        mask = rows != cols
        self.row = rows[mask]
        self.col = cols[mask]
        self.val = np.clip(vals[mask], 0.0, 1.0)

    def _find_gt(self) -> str:
        path = os.path.join(self.scene_dir, "gt.npz")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"missing gt npz in {self.scene_dir}; tried gt.npz")

    def _read_npz_field(self, data, candidates: List[str], dtype) -> np.ndarray:
        for key in candidates:
            if key in data:
                return data[key].astype(dtype)
        raise KeyError(f"Missing fields {candidates} in {self.gt_npz}")
    
    def _load_image_names(self, npz_data) -> List[str]:
        # Strategy 1: image_index.json (Most reliable for GL3D)
        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
            # Handle list ["a.jpg", ...] or dict {"name": id}
            if isinstance(index, list):
                return [str(x) for x in index]
            if "id_to_name" in index:
                return [str(x) for x in index["id_to_name"]]
            if "name_to_id" in index:
                return sorted(index["name_to_id"], key=index["name_to_id"].get)

        # Strategy 2: Embedded in NPZ
        for key in ["images", "filenames"]:
            if key in npz_data:
                return [str(x) for x in npz_data[key]]

        # Strategy 3: List directory (Fallback)
        if os.path.exists(self.images_dir):
            return sorted([x for x in os.listdir(self.images_dir) if x.lower().endswith(('.jpg', '.png'))])
            
        raise FileNotFoundError(f"Could not determine image order in {self.scene_dir}")

    def dense_overlap(self, idx: np.ndarray, add_self: bool = True) -> np.ndarray:
        """Extract dense subgraph overlaps.

        Args:
            idx (np.ndarray): Node indices, shape (n,).
            add_self (bool): If True, fill diagonal with 1.

        Returns:
            np.ndarray: Dense overlap matrix O of shape (n, n), dtype float32.
        """
        n = idx.shape[0]
        map_idx = -np.ones(self.N, dtype=np.int64)
        map_idx[idx] = np.arange(n, dtype=np.int64)

        mu = map_idx[self.row]
        mv = map_idx[self.col]
        sel = (mu >= 0) & (mv >= 0)
        mu, mv = mu[sel], mv[sel]
        w = self.val[sel]

        O = np.zeros((n, n), dtype=np.float32)
        if mu.size > 0:
            # Max to guard against duplicated edges
            O[mu, mv] = np.maximum(O[mu, mv], w)
        if add_self:
            np.fill_diagonal(O, 1.0)
        return O

    def neighbor_lists(self, iou_th: float = 0.2) -> List[List[int]]:
        """Adjacency as neighbor lists under IoU threshold.

        Args:
            iou_th (float): IoU threshold.

        Returns:
            List[List[int]]: For each node u, list of neighbors v.
        """
        adj: List[List[int]] = [[] for _ in range(self.N)]
        valid_edges = self.val >= iou_th
        for u, v in zip(self.row[valid_edges], self.col[valid_edges]):
            adj[int(u)].append(int(v))
        return adj

    def fetch_nodes(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch all nodes and their overlap weights with node idx."""
        mask = self.row == idx
        ov_map = {int(n): float(w) for n, w in zip(self.col[mask], self.val[mask])}
        
        all_nodes = np.delete(np.arange(self.N, dtype=np.int64), idx)
        weights = np.array([ov_map.get(int(n), 0.0) for n in all_nodes], dtype=np.float32)
        
        return all_nodes, weights