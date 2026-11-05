"""Read static per-scene image identities and sparse overlap annotations.

Required: scene_dir/gt.npz with COO row/rows, col/cols and weight/val/vals arrays.
Image order comes from image_index.json first (a list, id_to_name list or
name_to_id mapping), then the NPZ images/filenames field, then sorted filenames
in images/. Indices address this exact order; changing it changes GT identities.
Image roots may be supplied independently from the annotation directory.

Weights are clipped to [0,1], self loops removed and undirected duplicates use max.
Dense subgraph construction preserves the existing missing-edge-as-zero rule;
this module does not infer relationships from current embeddings or class IDs.
"""

import json
import os
from typing import List, Optional, Tuple
import numpy as np

from ..identities import integer_tensor


def _image_names(values):
    """Preserve filename identities without coercing malformed metadata."""
    if isinstance(values, (str, bytes, dict)):
        raise TypeError("Image names must be a sequence of strings")
    names = list(values)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Image names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Image names must be unique")
    return names


class SceneGraph:
    """Expose stable image paths and static COO overlap for one scene.

    scene_dir identifies the annotation source; node indices are scene-local
    physical image identities. No optimizer, Trainer or online queue is owned.
    """

    def __init__(self, scene_dir: str, image_root: Optional[str] = None):
        self.scene_dir = scene_dir
        if image_root is not None:
            if not os.path.isdir(image_root):
                raise FileNotFoundError(f"image root does not exist: {image_root}")
            self.images_dir = image_root
        else:
            images_dir = os.path.join(scene_dir, "images")
            self.images_dir = images_dir if os.path.exists(images_dir) else scene_dir
        self.image_index_file = os.path.join(scene_dir, "image_index.json")
        self.gt_npz = self._find_gt()

        # Load Metadata and Overlap Matrix
        with np.load(self.gt_npz, allow_pickle=True) as data:
            # 1. Load Image Names
            self.image_names = self._load_image_names(data)
            self.image_paths = [
                os.path.join(self.images_dir, name) for name in self.image_names
            ]
            self.N = len(self.image_names)

            row = self._read_node_indices(data, ["row", "rows"])
            col = self._read_node_indices(data, ["col", "cols"])
            val = self._read_npz_field(data, ["weight", "val", "vals"]).astype(
                np.float32
            )
        if row.ndim != 1 or col.shape != row.shape or val.shape != row.shape:
            raise ValueError("COO row, col and weight must be aligned vectors")
        if not np.isfinite(val).all():
            raise ValueError("Scene overlap weights must be finite")

        rows, cols = np.concatenate([row, col]), np.concatenate([col, row])
        values = np.clip(np.concatenate([val, val]), 0.0, 1.0)
        keep = rows != cols
        rows, cols, values = rows[keep], cols[keep], values[keep]
        # Stable first-occurrence order preserves neighbor order on unique data.
        # Both directions share a max after symmetrization.
        _, first, inverse = np.unique(
            np.stack((rows, cols), axis=1),
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        order = np.argsort(first)
        maxima = np.full(len(first), -np.inf, dtype=np.float32)
        np.maximum.at(maxima, inverse, values)
        self.row, self.col = rows[first[order]], cols[first[order]]
        self.val = maxima[order]

        # Keep annotation-record exposure separate from the canonical GT view.
        self._annotation_order = None
        self._annotation_weights = None
        if len(first) != len(rows):
            canonical_index = np.empty_like(order)
            canonical_index[order] = np.arange(len(order))
            self._annotation_order = canonical_index[inverse]
            if not np.array_equal(values, self.val[self._annotation_order]):
                self._annotation_weights = values

    def _find_gt(self) -> str:
        path = os.path.join(self.scene_dir, "gt.npz")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"missing gt npz in {self.scene_dir}; tried gt.npz")

    def _read_npz_field(self, data, candidates: List[str]) -> np.ndarray:
        for key in candidates:
            if key in data:
                return data[key]
        raise KeyError(f"Missing fields {candidates} in {self.gt_npz}")

    def _read_node_indices(self, data, names):
        return integer_tensor(
            self._read_npz_field(data, names),
            name="Scene node indices",
            minimum=0,
            maximum=self.N - 1,
        ).numpy()

    def _load_image_names(self, npz_data) -> List[str]:
        # Prefer an explicit node order over filename discovery.
        if os.path.exists(self.image_index_file):
            with open(self.image_index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
            # Handle list ["a.jpg", ...] or dict {"name": id}
            if isinstance(index, list):
                return _image_names(index)
            if "id_to_name" in index:
                return _image_names(index["id_to_name"])
            if "name_to_id" in index:
                mapping = index["name_to_id"]
                positions = integer_tensor(
                    list(mapping.values()),
                    name="Image-index positions",
                    minimum=0,
                    maximum=len(mapping) - 1,
                ).numpy()
                if len(np.unique(positions)) != len(mapping):
                    raise ValueError("Image-index positions must be unique")
                names = list(mapping)
                return _image_names([names[i] for i in np.argsort(positions)])

        # Strategy 2: Embedded in NPZ
        for key in ["images", "filenames"]:
            if key in npz_data:
                return _image_names(npz_data[key])

        # Strategy 3: List directory (Fallback)
        if os.path.exists(self.images_dir):
            return sorted(
                [
                    x
                    for x in os.listdir(self.images_dir)
                    if x.lower().endswith((".jpg", ".png"))
                    and os.path.isfile(os.path.join(self.images_dir, x))
                ]
            )

        raise FileNotFoundError(f"Could not determine image order in {self.scene_dir}")

    def dense_overlap(self, idx: np.ndarray, add_self: bool = True) -> np.ndarray:
        """Extract dense subgraph overlaps.

        Args:
            idx (np.ndarray): Node indices, shape (n,).
            add_self (bool): If True, fill diagonal with 1.

        Returns:
            np.ndarray: Dense overlap matrix O of shape (n, n), dtype float32.
            Repeated indices repeat the same physical node's row and column.
        """
        idx = integer_tensor(
            idx, name="Subgraph node indices", minimum=0, maximum=self.N - 1
        ).numpy()
        if idx.ndim != 1:
            raise ValueError("Subgraph node indices must be a vector")
        unique_nodes, inverse = np.unique(idx, return_inverse=True)
        count = len(unique_nodes)
        local_positions = -np.ones(self.N, dtype=np.int64)
        local_positions[unique_nodes] = np.arange(count, dtype=np.int64)
        source, target = local_positions[self.row], local_positions[self.col]
        selected = (source >= 0) & (target >= 0)
        overlap = np.zeros((count, count), dtype=np.float32)
        overlap[source[selected], target[selected]] = self.val[selected]
        if add_self:
            np.fill_diagonal(overlap, 1.0)
        return overlap[inverse[:, None], inverse[None, :]]

    def neighbor_lists(
        self, iou_th: float = 0.2, *, preserve_multiplicity: bool = False
    ) -> List[List[int]]:
        """Return unique thresholded graph neighbors, in first-occurrence order.

        preserve_multiplicity=True exposes original directed annotation records
        (including their original clipped weights) for the existing anchor-expand
        sampling protocol. Repeated records affect exposure, never canonical GT.
        """
        row, col, values = self.row, self.col, self.val
        if preserve_multiplicity and self._annotation_order is not None:
            row, col = row[self._annotation_order], col[self._annotation_order]
            values = (
                values[self._annotation_order]
                if self._annotation_weights is None
                else self._annotation_weights
            )
        adjacency: List[List[int]] = [[] for _ in range(self.N)]
        selected = values >= iou_th
        for source, target in zip(row[selected], col[selected]):
            adjacency[int(source)].append(int(target))
        return adjacency

    def fetch_nodes(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch all nodes and their overlap weights with node idx."""
        index = integer_tensor(
            idx, name="Scene node index", minimum=0, maximum=self.N - 1
        )
        if index.ndim != 0:
            raise ValueError("Scene node index must be scalar")
        idx = int(index)
        mask = self.row == idx
        ov_map = {int(n): float(w) for n, w in zip(self.col[mask], self.val[mask])}

        all_nodes = np.delete(np.arange(self.N, dtype=np.int64), idx)
        weights = np.array(
            [ov_map.get(int(n), 0.0) for n in all_nodes], dtype=np.float32
        )

        return all_nodes, weights
