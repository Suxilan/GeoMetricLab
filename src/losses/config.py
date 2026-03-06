from dataclasses import dataclass, field
from typing import Set, Dict, Any

try:
    from pytorch_metric_learning import losses as pml_losses
    from pytorch_metric_learning import miners as pml_miners
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "pytorch-metric-learning is required for GeoMetricLoss. Install via `pip install pytorch-metric-learning`."
    ) from exc

@dataclass
class LossConfig:
    tuple_len: int
    compatible_miners: Set[str] = field(default_factory=set)
    allow_explicit_indices: bool = True

# --- 工厂映射（统一在此处管理） ---
LOSS_FACTORIES: Dict[str, Any] = {
    "contrastive": pml_losses.ContrastiveLoss,
    "triplet": pml_losses.TripletMarginLoss,
    "tripletmargin": pml_losses.TripletMarginLoss,
    "multisimilarity": pml_losses.MultiSimilarityLoss,
    "multi_similarity": pml_losses.MultiSimilarityLoss,
    "instance": pml_losses.InstanceLoss,
    "ntxent": pml_losses.NTXentLoss,
    "supcontrastive": pml_losses.SupConLoss,
}

MINER_FACTORIES: Dict[str, Any] = {
    "batch_hard": pml_miners.BatchHardMiner,
    "batch_easy": pml_miners.BatchEasyHardMiner,
    "triplet_margin": pml_miners.TripletMarginMiner,
    "pair_margin": pml_miners.PairMarginMiner,
    "multi_similarity": pml_miners.MultiSimilarityMiner,
    "multisimilarity": pml_miners.MultiSimilarityMiner,
    "distance_weighted": pml_miners.DistanceWeightedMiner,
}

# --- 核心映射表 ---
# 这是一个“白名单”，不在名单里的组合直接报错
LOSS_REGISTRY = {
    # 1. Contrastive Loss (Pair-based)
    # 适合: GL3D (Pair模式)
    "contrastive": LossConfig(
        tuple_len=4,  # (a1, p, a2, n)
        compatible_miners={"pair_margin", "multi_similarity", "batch_easy"},
    ),

    # 2. Triplet Loss (Triplet-based)
    # 适合: GL3D (Triplet模式), University-1652
    "triplet": LossConfig(
        tuple_len=3,  # (a, p, n)
        compatible_miners={"batch_hard", "triplet_margin", "distance_weighted"},
    ),
    
    # 别名支持
    "tripletmargin": LossConfig(
        tuple_len=3, 
        compatible_miners={"batch_hard", "triplet_margin", "distance_weighted"}
    ),

    # 3. MultiSimilarity Loss (Pair-based internally)
    # 适合: University-1652 (配合 Miner)
    # GL3D: 如果 dataset 产生 pair tuple 也可以用
    "multisimilarity": LossConfig(
        tuple_len=4, 
        compatible_miners={"multi_similarity", "pair_margin", "batch_easy"},
    ),
    "multi_similarity": LossConfig(
        tuple_len=4,
        compatible_miners={"multi_similarity", "pair_margin", "batch_easy"},
    ),
    
    # 4. NTXent Loss (Self-Supervised / Pair)
    # 通常自带内部挖掘逻辑，一般不外挂 Miner
    "ntxent": LossConfig(
        tuple_len=4,
        compatible_miners=set(), # 通常不需要外部 miner
    ),
    "instance": LossConfig(
        tuple_len=4,
        compatible_miners=set(),
    ),

    # 5. SupCon Loss (Supervised Contrastive)
    "supcontrastive": LossConfig(
        tuple_len=4,
        compatible_miners=set(),
    ),
}
# Miner 输出维度定义 (必须与 Loss 需求一致)
MINER_OUTPUT_SHAPE = {
    "batch_hard": 3,
    "batch_easy": 4,
    "triplet_margin": 3,
    "pair_margin": 4,
    "multi_similarity": 4,
    "distance_weighted": 3,
}