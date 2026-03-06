"""Collate functions for subgraph datasets"""

from typing import List, Dict, Optional
import torch


def make_pad_collate(diag_weight: Optional[float] = None):
    """创建带填充的collate函数工厂
    
    Args:
        diag_weight (Optional[float]): 如果为None，不返回pair_weight；
            如果为float x，返回'pair_weight'，将对角线设置为x（其他为1）
    
    Returns:
        Callable: 将样本列表映射到填充batch字典的collate函数
    """

    def pad_collate(batch: List[Dict]) -> Dict:
        B = len(batch)
        n_max = max(x["num_nodes"] for x in batch)

        images = None
        if batch[0]["images"] is not None:
            C, H, W = batch[0]["images"].shape[1:]
            images = torch.zeros(B, n_max, C, H, W)

        O = torch.zeros(B, n_max, n_max)
        M_n = torch.zeros(B, n_max, dtype=torch.bool)
        M_p = torch.zeros(B, n_max, n_max, dtype=torch.bool)
        P_w = None
        if diag_weight is not None:
            P_w = torch.ones(B, n_max, n_max, dtype=torch.float32)

        scene_ids: List[str] = []
        image_paths: List[List[str]] = []

        for b, item in enumerate(batch):
            n = item["num_nodes"]
            scene_ids.append(item["scene_id"])
            image_paths.append(item["image_paths"])

            O[b, :n, :n] = item["overlap"]
            M_n[b, :n] = True
            M_p[b, :n, :n] = True

            if images is not None:
                images[b, :n] = item["images"]
            if P_w is not None:
                P_w[b, :n, :n].fill_(1.0)
                diag_idx = torch.arange(n)
                P_w[b, diag_idx, diag_idx] = float(diag_weight)

        out = {
            "scene_ids": scene_ids,
            "image_paths": image_paths,
            "overlap": O,  # (B, n_max, n_max)
            "node_mask": M_n,  # (B, n_max)
            "pair_mask": M_p,  # (B, n_max, n_max)
            "images": images,  # (B, n_max, 3, H, W) or None
        }
        if P_w is not None:
            out["pair_weight"] = P_w  # (B, n_max, n_max)
        return out

    return pad_collate

