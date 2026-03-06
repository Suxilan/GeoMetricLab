from typing import List, Dict, Union
from rich.console import Console
from rich.table import Table
from rich import box

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from typing import Union, List, Dict

LARGE_SCALE_THRESHOLD = 4000
def compute_metrics(
    descriptors: Union[np.ndarray, torch.Tensor],
    num_references: int,
    num_queries: int,
    ground_truth: List[List[int]],
    k_values: List[int] = [1, 5, 10],
    metric: str = "cosine",
    exclude_self: bool = False,
) -> Dict[str, float]:
    """Compute R@k, CMC@k, mAP@k and global mAP."""
    if isinstance(descriptors, np.ndarray):
        descriptors = torch.from_numpy(descriptors)

    total = descriptors.shape[0]
    assert num_references + num_queries == total, (
        "Number of references and queries do not match the number of descriptors."
    )
    assert len(ground_truth) == num_queries, "len(ground_truth) must equal num_queries"
    
    if exclude_self:
        assert num_references == num_queries, (
            "When exclude_self is True, num_references must equal num_queries."
        )
    if metric == "cosine":
        descriptors = F.normalize(descriptors, p=2, dim=-1)

    if torch.cuda.is_available():
        descriptors = descriptors.cuda()

    db_feats = descriptors[:num_references]
    q_feats = descriptors[num_references:]

    # sanity check: GT index范围
    max_gt = max((max(gt) for gt in ground_truth if len(gt) > 0), default=-1)
    if max_gt >= num_references:
        raise ValueError("Ground truth index out of range of reference descriptors.")

    search_k = num_references
    use_faiss = num_references > LARGE_SCALE_THRESHOLD

    if use_faiss:
        indices = _search_faiss(db_feats, q_feats, search_k, metric)
    else:
        indices = _search_pytorch(db_feats, q_feats, search_k, metric)

    return _calculate_stats(
        indices,
        ground_truth,
        k_values,
        exclude_self=exclude_self
    )


def _search_pytorch(db: torch.Tensor, q: torch.Tensor, k: int, metric: str) -> torch.Tensor:
    """Brute-force search in PyTorch."""
    device = db.device
    q = q.to(device)

    if metric == "cosine":
        sim = torch.mm(q, db.t())          # [Q, N]
        _, indices = torch.topk(sim, k=k, dim=1, largest=True)
    else:  # L2
        dist = torch.cdist(q, db, p=2)     # [Q, N]
        _, indices = torch.topk(dist, k=k, dim=1, largest=False)

    return indices.cpu()


def _search_faiss(db: torch.Tensor, q: torch.Tensor, k: int, metric: str) -> torch.Tensor:
    """FAISS CPU search."""
    d = db.shape[1]
    db_np = db.detach().cpu().numpy()
    q_np = q.detach().cpu().numpy()

    # normalize for cosine if requested
    if metric == "cosine":
        faiss.normalize_L2(db_np)
        faiss.normalize_L2(q_np)
        cpu_index = faiss.IndexFlatIP(d)
    else:
        cpu_index = faiss.IndexFlatL2(d)

    # Prefer GPU index when available; fall back to CPU on any failure
    try:
        # initialize GPU resources and transfer CPU index to GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(db_np)
        _, indices = gpu_index.search(q_np, k)
    except Exception:
        # fallback to CPU index
        cpu_index.add(db_np)
        _, indices = cpu_index.search(q_np, k)

    return torch.from_numpy(indices)


def _calculate_stats(
    predictions: torch.Tensor,
    ground_truth: List[List[int]],
    k_values: List[int],
    exclude_self: bool = False,
) -> Dict[str, float]:
    """Core retrieval statistics."""
    preds = predictions.cpu().numpy()
    num_queries_total = preds.shape[0]

    k_values = sorted(set(k_values))
    gt_sets = [set(g) for g in ground_truth]

    valid_queries = 0
    sum_map_global = 0.0
    sum_map_at_k = {k: 0.0 for k in k_values}
    sum_recall_at_k = {k: 0.0 for k in k_values}
    sum_cmc_at_k = {k: 0.0 for k in k_values}

    for q_idx in range(num_queries_total):
        true_ids = gt_sets[q_idx]
        if not true_ids:
            continue

        valid_queries += 1
        num_pos = len(true_ids)

        # 原始 ranking（按 reference 索引）
        ranking = preds[q_idx].tolist()  # convert to python list for filtering

        ranks = []
        precisions = []
        hits = 0

        current_rank = 0
        for db_idx in ranking:
            # --- [关键逻辑] 排除自己 ---
            # 前提：Query i 对应 DB i (GL3D/ReID Self-Retrieval 常用设定)
            if exclude_self and db_idx == q_idx:
                continue
            
            # 仅当不是自己时，Rank 才 +1
            current_rank += 1
            
            if db_idx in true_ids:
                hits += 1
                ranks.append(current_rank)
                
                # Precision at this rank
                precisions.append(hits / current_rank)
                
                # 优化：找齐了就退出当前 Query 的循环
                if hits == num_pos:
                    break

        if not ranks:  # 理论上不会发生（因为 true_ids 非空），防御而已
            continue

        # 全局 AP
        ap_global = sum(precisions) / num_pos
        sum_map_global += ap_global

        # 各个 K 的 AP@k / R@k / CMC@k
        for k in k_values:
            hits_in_k_idx = [i for i, r in enumerate(ranks) if r <= k]
            num_hits_in_k = len(hits_in_k_idx)

            if num_hits_in_k > 0:
                prec_sum_k = sum(precisions[i] for i in hits_in_k_idx)
                ap_k = prec_sum_k / num_pos
            else:
                ap_k = 0.0

            sum_map_at_k[k] += ap_k
            sum_recall_at_k[k] += num_hits_in_k / num_pos
            if ranks[0] <= k:
                sum_cmc_at_k[k] += 1.0

    print(f"[compute_metrics] valid queries: {valid_queries}/{num_queries_total}")

    metrics: Dict[str, float] = {}
    if valid_queries == 0:
        metrics["mAP"] = 0.0
        for k in k_values:
            metrics[f"mAP@{k}"] = 0.0
            metrics[f"R@{k}"] = 0.0
            metrics[f"CMC@{k}"] = 0.0
    else:
        metrics["mAP"] = (sum_map_global / valid_queries) * 100.0
        for k in k_values:
            metrics[f"mAP@{k}"] = (sum_map_at_k[k] / valid_queries) * 100.0
            metrics[f"R@{k}"] = (sum_recall_at_k[k] / valid_queries) * 100.0
            metrics[f"CMC@{k}"] = (sum_cmc_at_k[k] / valid_queries) * 100.0

    metrics["num_valid_queries"] = float(valid_queries)
    metrics["num_queries_total"] = float(num_queries_total)
    return metrics


def display_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """使用 Rich 动态打印指标，自动分组 R@k, mAP@k, CMC@k，并将 Global mAP 单独显示。"""
    console = Console()
    
    # 1. 定义排序键
    def sort_key(key):
        # 将 Global mAP 放在最后显示
        if key == "mAP":
            return ("~Global_mAP", 0)
        
        parts = key.split('@')
        name = parts[0]
        # 提取 K 值进行数字排序，否则默认 -1
        k_val = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
        return (name, k_val)

    sorted_keys = sorted(metrics.keys(), key=sort_key)

    # 2. 创建表格
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Metric", style="magenta")
    table.add_column("Value (%)", justify="right", style="green")

    # 3. 遍历并智能添加分隔线
    last_group = None
    
    for key in sorted_keys:
        # 定义分组逻辑：Global mAP 视为独立组，其他按 @ 前缀分组
        if key == "mAP":
            current_group = "Global_Summary"
        elif "@" in key:
            current_group = key.split('@')[0]
        else:
            current_group = key

        # 组别变化时添加分隔线 (首行除外)
        if last_group is not None and current_group != last_group:
            table.add_section()
        
        table.add_row(key, f"{metrics[key]:.2f}")
        last_group = current_group

    console.print(table)