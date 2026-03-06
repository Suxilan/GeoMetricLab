import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.metrics import compute_metrics, display_metrics
from src.utils.io import gather_distributed_predictions
from src.utils.io import save_features_h5, load_features_h5
from src.utils.logger import print_rank_0
from src.rerank.rerankers import apply_reranking

from src.datasets import University1652Dataset
from src.models.geoencoder import GeoEncoder

from config.model_hubs import MODEL_HUBS
from config.transform_hubs import get_transform
from weights.util import get_weight_config, load_weights


class University1652EvalModule(pl.LightningModule):
    """Simple Lightning Module for University1652 evaluation - only contains predict_step.

    Supports optional TTA (scales + horizontal flip) controlled at init.
    """

    def __init__(self, model: nn.Module, tta_scales: Optional[List[float]] = None, tta_flip: bool = False):
        super().__init__()
        self.model = model
        # TTA settings
        self.tta_scales = tta_scales if tta_scales is not None else [1.0]
        self.tta_flip = bool(tta_flip)

    # --- TTA helper -------------------------------------------------------
    def _forward_tta(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply optional flip + multi-scale TTA and return aggregated, normalized features."""
        aug_images: List[torch.Tensor] = []
        for scale in self.tta_scales:
            if scale == 1.0:
                aug_images.append(x)
            else:
                aug_images.append(F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False))
            if self.tta_flip:
                flipped = torch.flip(aug_images[-1], dims=[3])
                aug_images.append(flipped)

        feats_sum: Optional[torch.Tensor] = None
        for img in aug_images:
            y = model(img)
            f = y[0] if isinstance(y, tuple) else y
            feats_sum = f if feats_sum is None else feats_sum + f

        feats_sum = feats_sum / len(aug_images)
        return F.normalize(feats_sum, p=2, dim=-1)

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Extract features for evaluation, using TTA if configured."""
        images, indices = batch

        with torch.no_grad():
            if (self.tta_scales and (len(self.tta_scales) > 1)) or self.tta_flip:
                features = self._forward_tta(self.model, images)
            else:
                outputs = self.model(images)
                features = outputs[0] if isinstance(outputs, tuple) else outputs
            features = features.float()

        return {
            "features": features.detach().cpu(),  # (B, D)
            "indices": indices.cpu(),             # (B,)
        }

def load_model_and_weights(args) -> nn.Module:
    # Determine weight path and config
    weight_path = None
    model_config = None

    if args.model is not None:
        if args.model in MODEL_HUBS:
            model_config = MODEL_HUBS[args.model]
        else:
            raise SystemExit(f"Error: No model config found for '{args.model}'. Available models: {list(MODEL_HUBS.keys())}")
        
    if args.weights is not None:
        # Resolve weight path
        weight_path, model_config = get_weight_config(args.weights)

        if model_config is None:
            raise SystemExit(f"Error: No config found for {args.weights}.")

        weight_path = weight_path or model_config.get("weight_path")
        if weight_path is None:
            raise SystemExit(f"Hub entry '{args.weights}' has no weight_path; cannot evaluate without weights.")
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise SystemExit(f"Weight file not found for '{args.weights}': {weight_path}")
        
    if model_config is None:
        raise SystemExit("Error: No model config provided. Use --model or --weights to specify model configuration.")
    
    print_rank_0(f"Model Config: {model_config}")
    print_rank_0(f"Weights: {weight_path}")
    
    # Setup model
    model = GeoEncoder(
        backbone_name=model_config['backbone'],
        aggregator_name=model_config['aggregator'],
        backbone_args=model_config.get('backbone_args', {}),
        aggregator_args=model_config.get('aggregator_args', {}),
        whitening=model_config.get("whitening", False),
        whitening_dim=model_config.get("whitening_dim", 256),
    )
    
    # Load weights
    if weight_path:
        print_rank_0(f"Loading weights...")
        load_weights(model, str(weight_path), args.weights)

    agg = model.aggregator  # 别名，为了代码简洁
    h5_path = Path("u1652/feature_maps_dinov2b14_avg_10000.h5")
    print_rank_0(f"[*] Init NetVLAD from {h5_path}...")
    import faiss

    data = load_features_h5(h5_path)
    feats = data.get("features", None)

    # 展平: (N, C, H, W) -> (N*H*W, C)
    N, C, H, W = feats.shape
    descriptors = feats.transpose(0, 2, 3, 1).reshape(-1, C).astype(np.float32, copy=False)
    print_rank_0(f"   Loaded {descriptors.shape[0]} descriptors of dim {descriptors.shape[1]}.")

    # L2 归一化每个 descriptor（在 K-Means 前通常能提升聚类质量）
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    # 防止除以 0
    norms[norms == 0] = 1.0
    descriptors /= norms

    # 运行 K-Means
    kmeans = faiss.Kmeans(
        d=agg.in_channels, 
        k=agg.num_clusters, 
        niter=25, 
        verbose=True, 
        gpu=True,
        seed=42,
        max_points_per_centroid=16384,
    )
    kmeans.train(descriptors)

    # 应用结果 (自动计算 Alpha 和赋值权重)
    agg.init_from_clusters(kmeans.centroids, descriptors)


    return model, weight_path, model_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data/university1652/test", help='University1652 test dataset root (folder containing metadata)')
    parser.add_argument('--task', type=str, default="u1652_drone2satellite", help='Task name: u1652_drone2satellite or u1652_ground2satellite')
    parser.add_argument('--features', type=str, default=None, help='Path to pre-extracted features (.h5)')
    parser.add_argument('--save_cache', action='store_true', help='Save extracted features to cache (h5)')
    parser.add_argument('--reranking', type=str, default='none', help='Reranking method (none|gnn)')
    parser.add_argument('--model', type=str, default=None, help="Model config")
    parser.add_argument('--weights', type=str, default=None, help="Path to model weights file (relative to weights/ or absolute)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--devices', type=str, default='auto' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--strategy', type=str, default='auto' if torch.cuda.device_count() > 1 else None)
    parser.add_argument('--output', type=str, default='./results/university1652')
    args = parser.parse_args()

    # enable SDP knobs if present (PyTorch ≥ 2.x)
    if torch.cuda.is_available():
        try:
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    # Prepare dataset
    ds_root = args.root
    task_name = args.task
    dataset = University1652Dataset(
        dataset_path=ds_root, 
        dataset_name=task_name,
        input_transform=get_transform("u1652", args.img_size, train=False))

    # If features provided, load and assume ordering matches dataset
    if args.features:
        _, weight_path, model_config = load_model_and_weights(args)
        feat_path = Path(args.features)
        if feat_path.suffix == '.h5' or feat_path.suffix == '.hdf5':
            data = load_features_h5(feat_path, cache_dir=Path('cache') / 'u1652')
            feats = data['features']
        else:
            feats = np.load(str(feat_path))
        if isinstance(feats, np.ndarray):
            sorted_features = torch.from_numpy(feats)
        else:
            sorted_features = feats
        # descriptors shape should be (N, D)
        print(f"Loaded features {sorted_features.shape}")
        trainer = None
    else:
        # 1. Build Model and Load Weights
        model, weight_path, model_config = load_model_and_weights(args)
        print_rank_0(f"Evaluating on University1652 test set...")

        # DataLoader for feature extraction
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False, 
            pin_memory=True
        )

        # 3. Setup Trainer (Inference Mode)
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=args.devices,
            strategy=args.strategy,
            logger=False,
            enable_progress_bar=True
        )
        eval_module = University1652EvalModule(
            model,
            tta_scales=[1.0], # [1.0, 0.7071, 1.4142]
            tta_flip=False
        )
        raw_predictions = trainer.predict(eval_module, dataloaders=dataloader)

        # 5. Gather Results
        features, indices = gather_distributed_predictions(raw_predictions, trainer)
        if trainer.is_global_zero:
            # Sort features according to dataset order
            order = torch.argsort(indices)
            sorted_features = features[order]

            # Optionally save cache (rank 0 only)
            if args.save_cache:
                # determine output filename
                model_name = Path(weight_path).stem if weight_path else args.model
                filename = f"{model_name}_{task_name}_features.h5"
                # image names from dataset
                image_names = [str(p) for p in dataset.image_paths[order.numpy()]]
                save_features_h5(sorted_features.numpy(), image_names, Path(filename), cache_dir=Path('cache') / 'u1652', num_references=int(dataset.num_references), num_queries=int(dataset.num_queries))
                print_rank_0(f"Saved feature cache to {Path('cache') / 'u1652' / filename}")

    # Apply optional reranking -> produce descriptors for metrics
    if trainer is None or (trainer is not None and trainer.is_global_zero):
        num_refs = dataset.num_references
        num_queries = dataset.num_queries
        gallery_feats = sorted_features[:num_refs]
        query_feats = sorted_features[num_refs:]

        if args.reranking and args.reranking.lower() != 'none':
            query_feats, gallery_feats = apply_reranking(args.reranking, query_feats, gallery_feats)
        descriptors_for_metrics = torch.cat([gallery_feats, query_feats], dim=0)
    else:
        descriptors_for_metrics = None

    # 6. Evaluation (Rank 0 Only)
    if trainer is None or trainer.is_global_zero:
        print("Sorting and computing metrics...")

        metrics = compute_metrics(
            descriptors=descriptors_for_metrics,
            num_references=dataset.num_references,
            num_queries=dataset.num_queries,
            ground_truth=dataset.ground_truth,
            k_values=[1, 5, 10],
            metric="cosine"
        )

        filtered_metrics = {
            k: v for k, v in metrics.items() 
            if "R" not in k and "num_queries_total" not in k and "num_valid_queries" not in k
        }

        # Display and save
        display_metrics(filtered_metrics, title=f"University1652 Evaluation-{task_name}")

        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        # If output is a directory or default, construct filename
        if out_path.is_dir():
            # Create results dir if not exists
            model_name = Path(weight_path).stem if weight_path else args.model
            filename = f"{model_name}" + ("" if args.reranking.lower() == 'none' else f"_{args.reranking.lower()}") + f"_{task_name}_eval.json"
            out_path = out_path / filename

        save_obj = {
            "weight_file": str(weight_path) if weight_path else "None",
            "config": model_config,
            "use_reranking": args.reranking,
            "img_size": args.img_size,
            'metrics': filtered_metrics,
            'num_references': int(dataset.num_references),
            'num_queries': int(dataset.num_queries),
            "num_valid_queries": int(metrics.get("num_valid_queries", 0))
        }

        with open(out_path, 'w') as f:
            json.dump(save_obj, f, indent=2)

        print(f"Saved results to {out_path}")

if __name__ == '__main__':
    main()

"""
Usage examples:
for static feature evaluation only:
python scripts/eval_university1652.py  --task u1652_drone2satellite \
      --weights resnet50_gem_instance_circle_u1652_pretrain 
      --features resnet50_gem_instance_circle_u1652_pretrain_features.h5
for inference + evaluation:
python scripts/eval_university1652.py  --task u1652_drone2satellite \
      --weights resnet50_gem_instance_circle_u1652_pretrain
      --save_cache # optional, save extracted features to cache

"""

