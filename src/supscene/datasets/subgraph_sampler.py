"""SubgraphSampler：子图采样器"""

import random
from typing import List, Dict
import numpy as np

from .scenegraph import SceneGraph


class SubgraphSampler:
    """子图采样器：支持uniform、anchor_expand、balanced三种模式
    
    Modes:
        - "uniform": 均匀采样，不考虑结构
        - "anchor_expand": 基于IoU的BFS扩展，从随机锚点开始
        - "balanced": 贪心优化，目标达到特定的正样本对比例
    """
    
    def __init__(
        self, 
        mode: str = "anchor_expand", 
        iou_th: float = 0.0, 
        topk_per_hop: int = 32, 
        max_hops: int = 3,
        balanced_small_graph: bool = True, 
        small_graph_threshold: int = 16,
        target_positive_ratio: float = 0.5
    ):
        assert mode in ("uniform", "anchor_expand", "balanced"), f"Unknown mode: {mode}"
        self.mode = mode
        self.iou_th = iou_th
        self.topk_per_hop = topk_per_hop
        self.max_hops = max_hops
        self.balanced_small_graph = balanced_small_graph
        self.small_graph_threshold = small_graph_threshold
        self.target_positive_ratio = target_positive_ratio

    def sample(self, G: SceneGraph, n_sub: int) -> np.ndarray:
        """从场景图G中采样大小为n_sub的子图
        
        Args:
            G (SceneGraph): 场景图
            n_sub (int): 要采样的节点数量
        
        Returns:
            np.ndarray: 节点索引，shape (n,), dtype int64
        """
        N = G.N
        if n_sub >= N:
            return np.arange(N, dtype=np.int64)

        if (
            self.balanced_small_graph
            and n_sub <= self.small_graph_threshold
            and self.mode in ["uniform", "balanced"]
        ):
            return self._balanced_sample(G, n_sub)

        if self.mode == "uniform":
            return np.random.choice(N, size=n_sub, replace=False)
        elif self.mode == "balanced":
            return self._balanced_sample(G, n_sub)

        # anchor expansion based on adjacency
        adj = G.neighbor_lists(self.iou_th)
        anchor = np.random.randint(0, N)
        visited = set([int(anchor)])
        frontier = [int(anchor)]
        hops = 0
        while len(visited) < n_sub and len(frontier) > 0 and hops < self.max_hops:
            new_frontier = []
            for u in frontier:
                nbrs = adj[u]
                if len(nbrs) == 0:
                    continue
                if len(nbrs) > self.topk_per_hop:
                    nbrs = random.sample(nbrs, self.topk_per_hop)
                for v in nbrs:
                    if v not in visited:
                        visited.add(int(v))
                        new_frontier.append(int(v))
                        if len(visited) >= n_sub:
                            break
                if len(visited) >= n_sub:
                    break
            frontier = new_frontier
            hops += 1

        if len(visited) < n_sub:
            rest = [x for x in range(N) if x not in visited]
            more = np.random.choice(rest, size=(n_sub - len(visited)), replace=False)
            visited.update(more.tolist())

        idx = np.fromiter(visited, dtype=np.int64)
        np.random.shuffle(idx)
        return idx[:n_sub]

    def _balanced_sample(self, G: SceneGraph, n_sub: int) -> np.ndarray:
        """贪心平衡采样：优化正/负样本对比例
        
        Args:
            G (SceneGraph): 场景图
            n_sub (int): 子图大小
        
        Returns:
            np.ndarray: 节点索引 (n,)
        """
        N = G.N
        if n_sub >= N:
            return np.arange(N, dtype=np.int64)

        # 预计算重叠字典用于快速查询
        O_dict: Dict[tuple, float] = {}
        for u, v, val in zip(G.row, G.col, G.val):
            if val >= self.iou_th:
                O_dict[(u, v)] = val
                O_dict[(v, u)] = val

        def has_o(i: int, j: int) -> bool:
            return (i, j) in O_dict

        def pos_ratio(nodes: List[int]) -> float:
            if len(nodes) < 2:
                return 0.0
            p, tot = 0, 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    tot += 1
                    if has_o(nodes[i], nodes[j]):
                        p += 1
            return p / tot if tot > 0 else 0.0

        nodes = np.random.choice(N, size=n_sub, replace=False).tolist()
        r = pos_ratio(nodes)

        T = min(50, N)  # max iterations
        tol = 0.05
        for _ in range(T):
            if abs(r - self.target_positive_ratio) <= tol:
                break
            need_more_pos = r < self.target_positive_ratio
            unused = [i for i in range(N) if i not in nodes]
            if not unused:
                break

            best_imp = 0.0
            best_swap = None
            lim = min(20, len(unused))
            for ridx in range(len(nodes)):
                for cand in unused[:lim]:
                    trial = nodes.copy()
                    trial[ridx] = cand
                    r_new = pos_ratio(trial)
                    imp = (r_new - r) if need_more_pos else (r - r_new)
                    if imp > best_imp:
                        best_imp, best_swap = imp, (ridx, cand)

            if best_swap and best_imp > 1e-2:
                ridx, cand = best_swap
                nodes[ridx] = cand
                r = pos_ratio(nodes)
            else:
                break

        idx = np.array(nodes, dtype=np.int64)
        np.random.shuffle(idx)
        return idx

