#!/usr/bin/env python3
"""
轻量版 GL3D 评估脚本（无需额外的 GL3DEvaluator）。

功能：
- 使用 `src/datasets/valid/gl3d.py` 中的 `GL3DDataset` 加载测试集
- 支持两种评估路径：
  1) 通过 `--features` 加载预先提取的特征 (.npy)
  2) 使用 `GeoEncoder` + 可选权重提取特征
- 使用 `src/utils/metrics.py` 中的 `compute_metrics` 计算 Recall@K 与 Global mAP

示例：
  python scripts/eval_gl3d_simple.py --features ./features.npy --output ./results.json
  python scripts/eval_gl3d_simple.py --weights ResNet50_512_cosplace.pth --img_size 512

"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import v2 as T2
# ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.metrics import compute_metrics, display_metrics
from src.utils.io import gather_distributed_predictions
from src.utils.io import save_features_h5, load_features_h5
from src.utils.logger import print_rank_0
from src.rerank.rerankers import apply_reranking

from src.datasets import GL3DDataset
from src.models.geoencoder import GeoEncoder

from config.model_hubs import MODEL_HUBS
from config.transform_hubs import get_transform
from weights.util import get_weight_config, load_weights

class GL3DEvalModule(pl.LightningModule):
    """Simple Lightning Module for GL3D evaluation - only contains predict_step."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """Extract features for evaluation."""
        images, indices = batch
        
        with torch.no_grad():
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

    return model, weight_path, model_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="./data/GL3D/test", help='GL3D test dataset root (folder containing metadata)')
    parser.add_argument('--features', type=str, default=None, help='Path to pre-extracted features (.h5)')
    parser.add_argument('--save_cache', action='store_true', help='Save extracted features to cache (h5)')
    parser.add_argument('--reranking', type=str, default='none', help='Reranking method (none|gnn)')
    parser.add_argument('--model', type=str, default=None, help="Model config")
    parser.add_argument('--weights', type=str, default=None, help="Path to model weights file (relative to weights/ or absolute)")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--devices', type=str, default='auto' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--strategy', type=str, default='auto' if torch.cuda.device_count() > 1 else None)
    parser.add_argument('--output', type=str, default='./results/gl3d')
    args = parser.parse_args()

    # Prepare dataset
    ds_root = args.root
    dataset = GL3DDataset(
        dataset_path=ds_root, 
        input_transform=get_transform("gl3d", args.img_size, train=False))

    # If features provided, load and assume ordering matches dataset
    if args.features:
        feat_path = Path(args.features)
        if feat_path.suffix == '.h5' or feat_path.suffix == '.hdf5':
            data = load_features_h5(feat_path, cache_dir=Path('cache') / 'GL3D')
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
        print_rank_0(f"Evaluating on GL3D test set...")
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
        eval_module = GL3DEvalModule(model)
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
                filename = f"{model_name}_features.h5"
                # image names from dataset
                image_names = [str(p) for p in dataset.image_paths[order.numpy()]]
                save_features_h5(sorted_features.numpy(), image_names, Path(filename), cache_dir=Path('cache') / 'GL3D', num_references=int(dataset.num_references), num_queries=int(dataset.num_queries))
                print_rank_0(f"Saved feature cache to {Path('cache') / 'GL3D' / filename}")

    # Apply optional reranking -> produce descriptors for metrics
    if trainer is None or (trainer is not None and trainer.is_global_zero):
        num_refs = dataset.num_references
        num_queries = dataset.num_queries
        gallery_feats = sorted_features[:num_refs]
        # For GL3D, queries share the same feature set
        query_feats = sorted_features[:num_queries]

        if args.reranking and args.reranking.lower() != 'none':
            query_feats, gallery_feats = apply_reranking(args.reranking, query_feats, gallery_feats)
        descriptors_for_metrics = torch.cat([gallery_feats, query_feats], dim=0)
    else:
        descriptors_for_metrics = None

    # 6. Evaluation (Rank 0 Only)
    if trainer is None or trainer.is_global_zero:
        print("Sorting and computing metrics...")
        
        # 直接调用 metrics.py 中的函数
        metrics = compute_metrics(
            descriptors=descriptors_for_metrics,
            num_references=dataset.num_references,
            num_queries=dataset.num_queries,
            ground_truth=dataset.ground_truth,
            k_values=[1, 25, 50, 100],
            metric="cosine",
            exclude_self=True,
        )

        filtered_metrics = {
            k: v for k, v in metrics.items() 
            if "CMC" not in k and "num_queries_total" not in k and "num_valid_queries" not in k
        }

        # Display and save
        display_metrics(filtered_metrics, title=f"GL3D Evaluation")

        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        # If output is a directory or default, construct filename
        if out_path.is_dir():
            model_name = Path(weight_path).stem if weight_path else args.model
            filename = f"{model_name}" + ("" if args.reranking.lower() == 'none' else f"_{args.reranking.lower()}") + f"_eval.json"
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


# use example:
# for static feature evaluation only:
# python scripts/eval_gl3d.py  
#       --weights resnet50_gem_instance_circle_u1652_pretrain 
#       --features resnet50_gem_instance_circle_u1652_pretrain_features.h5

# for inference + evaluation:
# python scripts/eval_gl3d.py  
#       --weights resnet50_gem_instance_circle_u1652_pretrain
#       --save_cache # optional, save extracted features to cache