"""Configuration generator CLI.

Generates YAML configuration files for training by combining canonical
configs from hubs (model, data, loss) with default settings.

Usage:
    python config/create_config.py --task train_gl3d --model resnet50_avg ...
"""

from __future__ import annotations
import argparse
import yaml
import os
import sys
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.model_hubs import MODEL_HUBS
from config.dataset_hubs import DATA_HUBS
from config.loss_hubs import LOSS_HUBS

# Default configurations for Trainer, Optimizer, Scheduler
_TRAINER_DEFAULT = {
    "max_epochs": 30,
    "precision": 32,
    "strategy": "ddp",
    "accelerator": "auto",
    "devices": "auto",
    "gradient_clip_val": 1.0,
    "log_every_n_steps": 10,
    "eval_enabled": False,
    "monitor": "eval/mAP",
    "checkpoint": {
        "mode": "max",
        "save_top_k": 3,
        "dirpath": "checkpoints/gl3d",
        "save_last": True,
    },
    "lr_monitor": True,
    "early_stopping": {
        "enabled": False,
        "mode": "min",
        "patience": 5,
        "min_delta": 0.0,
    },
    "eval": {
        "enabled": True,
        "every_n_epochs": 5,
        "root": "./data/GL3D",
        "split": "test",
        "batch_size": 64,
        "num_workers": 4,
        "mode": "similarity",
        "pos_th": 0.25,
    },
}

_OPTIMIZER_DEFAULT = {
    "name": "adamw",
    "lr": 1e-4,
    "weight_decay": 1e-4,
}

_SCHEDULER_DEFAULT = {
    "name": "cosine",
    "warmup_ratio": 0.1,
}


def create_config_dict(
    model_key: str,
    data_key: str,
    loss_key: str,
) -> Dict[str, Any]:
    """Generate a complete training configuration dictionary."""
    if model_key not in MODEL_HUBS:
        raise ValueError(f"Model key '{model_key}' not found. Available: {list(MODEL_HUBS.keys())}")
    if data_key not in DATA_HUBS:
        raise ValueError(f"Data key '{data_key}' not found. Available: {list(DATA_HUBS.keys())}")
    if loss_key not in LOSS_HUBS:
        raise ValueError(f"Loss key '{loss_key}' not found. Available: {list(LOSS_HUBS.keys())}")

    cfg = {
        "model": deepcopy(MODEL_HUBS[model_key]),
        "data": deepcopy(DATA_HUBS[data_key]),
        "loss": deepcopy(LOSS_HUBS[loss_key]),
        "trainer": deepcopy(_TRAINER_DEFAULT),
        "optimizer": deepcopy(_OPTIMIZER_DEFAULT),
        "scheduler": deepcopy(_SCHEDULER_DEFAULT),
    }
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Generate Training Config")
    parser.add_argument("--task", type=str, default="train_gl3d", help="Task name (defines output folder)")
    parser.add_argument("--model", type=str, default="resnet50_avg", help="Model hub key")
    parser.add_argument("--data", type=str, default="gl3d_default", help="Data hub key")
    parser.add_argument("--loss", type=str, default="multisimilarity_default", help="Loss hub key")
    parser.add_argument("--filename", type=str, default="default.yaml", help="Output filename")
    
    args = parser.parse_args()

    try:
        cfg = create_config_dict(args.model, args.data, args.loss)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine output path
    output_dir = Path("config") / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.filename

    print(f"Generating config for task '{args.task}'...")
    print(f"  Model: {args.model}")
    print(f"  Data:  {args.data}")
    print(f"  Loss:  {args.loss}")
    
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config saved to: {output_path}")


if __name__ == "__main__":
    main()
