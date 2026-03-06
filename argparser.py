from __future__ import annotations

import argparse
from typing import Any, Dict

from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoMetricLab training arguments")

    # General arguments
    parser.add_argument("--config", type=str, default="config/train_supscene_gl3d/default.yaml", help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--silent", action="store_true", help="Disable verbose output")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--dev", action="store_true", help="Enable fast dev run")
    parser.add_argument("--display_theme", type=str, help="Rich console theme")
    parser.add_argument("--use_proj", action="store_true", help="Enable projection head during training")
    parser.add_argument("--use_bn", action="store_true", help="Use batch normalization neck in the model")

    # Datamodule arguments
    parser.add_argument("--train_set", type=str, help="Training dataset name")
    parser.add_argument("--val_sets", nargs="+", help="Validation dataset names")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_workers", type=int, help="Dataloader workers")

    # Model/weights arguments
    parser.add_argument("--weights", type=str, help="Pretrained weights/config hub key")
    parser.add_argument("--model", type=str, help="Model config override key")

    # Trainer/optimizer overrides
    parser.add_argument("--max_epochs", type=int, help="Training epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--strategy", type=str, help="Strategy (ddp, auto, etc.)")
    parser.add_argument("--devices", type=str, help="Devices (auto, 1, 2, ...)")
    parser.add_argument("--grad_clip", type=float, help="Gradient clip value")
    parser.add_argument("--precision", type=str, help="Precision (16-mixed, 32, etc.)")

    # Checkpointing/Early stopping
    parser.add_argument("--save_top_k", type=int, default=None, help="Number of top checkpoints to save")
    parser.add_argument("--save_last", type=int, default=None, help="Whether to save last checkpoint")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume")

    args = parser.parse_args()

    # If a config file is provided, load it
    config = load_config(args.config) if args.config else {}

    # Update config with command-line arguments and default values
    config = update_config_with_args_and_defaults(config, args)

    return config


def update_config_with_args_and_defaults(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update the configuration dictionary with command-line arguments and default values.
    Priority: Command-line args > Config file values > Default values
    """
    default_config = {
        "experiment": "default",
        "seed": 42,
        "silent": False,
        "compile": False,
        "dev": False,
        "display_theme": "default",
        "weights": None,
        "model": None,
        "use_proj": False,
        "use_bn": False,
        "data": {
            "train_datapath": "data/GL3D/train",
            "train_set_name": "gl3d_subgraph",
            "train_image_size": [224, 224],
            "val_set_names": ["gl3d_test"],
            "val_image_size": [224, 224],
            "batch_size": 1,
            "num_workers": 4
        },

        "model": {
            "backbone": "resnet50",
            "aggregator": "gem",
            "backbone_args": {},
            "aggregator_args": {},
            "whitening": False,
            "whitening_dim": 1024,
        },

        "loss": {
            'module': 'src.supscene.losses',
            'class': 'SupConLoss',
            'params': {},
        },
        "optimizer": {
            "name": "adamw",
            "lr": 0.0001,
            "weight_decay": 0.0001,
        },

        "scheduler": {
            "warmup_start_factor": 0.0,
            "warmup_ratio": 0.1
        },
        "trainer": {
            "max_epochs": 50,
            "accelerator": "gpu",
            "devices": "auto",
            "strategy": "auto",
            "log_every_n_steps": 50,
            "gradient_clip_val": 0.0,
            "precision": 32,
            "sync_batchnorm": True,
            "check_val_every_n_epoch": 1,
            "monitor": "val/mAP",
            "checkpoint": {
                "save_last": True,
                "save_top_k": 3,
                "mode": "max"
            },
            "early_stopping": {
                "enabled": False,
                "patience": 5,
                "mode": "max",
                "min_delta": 0.0
            },
            "resume_from": None
        }

    }

    # Helper function to update nested dictionaries
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    # Update config with default values for missing keys
    config = update_nested_dict(default_config, config)

    # Update with command-line arguments if provided
    arg_dict = vars(args)

    # Uapdate general settings
    if arg_dict.get("seed") is not None:
        config["seed"] = arg_dict["seed"]
    if arg_dict.get("silent") is not None:
        config["silent"] = arg_dict["silent"]
    if arg_dict.get("compile") is not None:
        config["compile"] = arg_dict["compile"]
    if arg_dict.get("dev") is not None:
        config["dev"] = arg_dict["dev"]
    if arg_dict.get("display_theme") is not None:
        config["display_theme"] = arg_dict["display_theme"]
    if arg_dict.get("use_proj") is not None:
        config["use_proj"] = arg_dict["use_proj"]
    if arg_dict.get("use_bn") is not None:
        config["use_bn"] = arg_dict["use_bn"]
    
    # Update weights/model settings
    if arg_dict.get("weights") is not None:
        config["weights"] = arg_dict["weights"]
    if arg_dict.get("model") is not None:
        config["model"] = arg_dict["model"]
    
    # Update datamodule settings
    if arg_dict.get("train_set") is not None:
        config["data"]["train_set_name"] = arg_dict["train_set"]
    if arg_dict.get("val_sets") is not None:
        config["data"]["val_set_names"] = arg_dict["val_sets"]
    if arg_dict.get("batch_size") is not None:
        config["data"]["batch_size"] = arg_dict["batch_size"]
    if arg_dict.get("num_workers") is not None:
        config["data"]["num_workers"] = arg_dict["num_workers"]

    # Update trainer settings
    if arg_dict.get("max_epochs") is not None:
        config["trainer"]["max_epochs"] = arg_dict["max_epochs"]
    if arg_dict.get("lr") is not None:
        config["optimizer"]["lr"] = arg_dict["lr"]
    if arg_dict.get("strategy") is not None:
        config["trainer"]["strategy"] = arg_dict["strategy"]
    if arg_dict.get("devices") is not None:
        config["trainer"]["devices"] = arg_dict["devices"]
    if arg_dict.get("grad_clip") is not None:
        config["trainer"]["gradient_clip_val"] = arg_dict["grad_clip"]
    if arg_dict.get("precision") is not None:
        config["trainer"]["precision"] = arg_dict["precision"]

    # Uapdate checkpointing/early stopping
    if arg_dict.get("save_top_k") is not None:
        config["trainer"]["checkpoint"]["save_top_k"] = arg_dict["save_top_k"]
    if arg_dict.get("save_last") is not None:
        config["trainer"]["checkpoint"]["save_last"] = arg_dict["save_last"]
    if arg_dict.get("patience") is not None:
        config["trainer"]["early_stopping"]["patience"] = arg_dict["patience"]
    if arg_dict.get("resume_from") is not None:
        config["trainer"]["resume_from"] = arg_dict["resume_from"]

    return config

if __name__ == "__main__":
    config = parse_args()
    import pprint
    pprint.pprint(config)
