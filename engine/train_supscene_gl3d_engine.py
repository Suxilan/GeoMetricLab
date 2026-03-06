"""Training engine: SupScene on GL3D (new pipeline)

Usage:
    python engine/train_supscene_gl3d_engine.py \
        --config config/train_supscene_gl3d/default.yaml \
        --weights ResNet50_512_cosplace.pth
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

# project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import print_rank_0
from weights.util import get_weight_config, load_weights
from config.model_hubs import MODEL_HUBS
from src.pipeline import SupSceneDataModule, SupSceneFramework
from src.models.geoencoder import GeoEncoder
from src.utils.callbacks import (
    CustomRichModelSummary,
    CustomRichProgressBar,
    DatamoduleSummary,
    ModelFrameworkSummary,
)

def build_datamodule(cfg: Dict[str, Any]) -> SupSceneDataModule:
    data_cfg = cfg.get("data", {})
    train_size = tuple(data_cfg.get("train_image_size", (224, 224)))
    val_size = tuple(data_cfg.get("val_image_size", train_size))

    dm = SupSceneDataModule(
        train_datapath=data_cfg.get("train_datapath"),
        train_set_name=data_cfg.get("train_set_name"),
        train_image_size=train_size,
        batch_size=data_cfg.get("batch_size", 1),
        num_workers=data_cfg.get("num_workers", 4),
        n_sub=data_cfg.get("n_sub", 128),
        iou_th=data_cfg.get("iou_th", 0.2),
        scenes_per_epoch=data_cfg.get("scenes_per_epoch"),
        samples_per_scene=data_cfg.get("samples_per_scene"),
        adaptive_sampling=data_cfg.get("adaptive_sampling", True),
        min_images_per_scene=data_cfg.get("min_images_per_scene", 50),
        collate_diag_weight=data_cfg.get("collate_diag_weight"),
        batch_sampler=None,
        val_set_names=data_cfg.get("val_set_names", []),
        val_image_size=val_size,
    )
    return dm

def _normalize_model_cfg(model_entry: Any) -> Dict[str, Any]:
    if isinstance(model_entry, str):
        if model_entry in MODEL_HUBS:
            return MODEL_HUBS[model_entry].copy()
        raise SystemExit(f"Unknown model hub key '{model_entry}'. Available: {list(MODEL_HUBS.keys())}")
    if isinstance(model_entry, dict):
        return model_entry.copy()
    raise ValueError("model configuration must be a dict or hub key string")


def _merge_model_cfg(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def build_model(cfg: Dict[str, Any]):
    model_cfg = _normalize_model_cfg(cfg.get("model", {}))
    weights_key = cfg.get("weights")
    weight_path: Path | None = None

    if weights_key:
        weight_path, hub_cfg = get_weight_config(weights_key)
        if hub_cfg is None:
            raise SystemExit(f"Error: No config found for {weights_key}.")
        # model_cfg = _merge_model_cfg(model_cfg, hub_cfg)
        if weight_path is None:
            hub_path = hub_cfg.get("weight_path")
            if hub_path:
                weight_path = Path(hub_path)
        if weight_path is None:
            raise SystemExit(f"Hub entry '{weights_key}' has no weight_path.")
        if not weight_path.exists():
            raise SystemExit(f"Weight file not found: {weight_path}")

    model = GeoEncoder(
        backbone_name=model_cfg["backbone"],
        aggregator_name=model_cfg["aggregator"],
        backbone_args=model_cfg.get("backbone_args", {}),
        aggregator_args=model_cfg.get("aggregator_args", {}),
        whitening=model_cfg.get("whitening", False),
        whitening_dim=model_cfg.get("whitening_dim", 256),
    )

    if weight_path:
        print_rank_0(f"Loading weights from {weight_path}...")
        load_weights(model, str(weight_path), weights_key)

    return model


def build_criterion(cfg: Dict[str, Any]):
    loss_cfg = cfg.get("loss", {})
    module_name = loss_cfg.get("module")
    class_name = loss_cfg.get("class")
    params = loss_cfg.get("params", {}) or {}

    if not module_name or not class_name:
        raise ValueError("Loss configuration must include 'module' and 'class' entries.")

    module = importlib.import_module(module_name)
    loss_cls = getattr(module, class_name)
    return loss_cls(**params)


def build_trainer(cfg: Dict[str, Any], monitor_key: str, experiment_name: str):
    train_cfg = cfg["trainer"]

    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    ckpt_cfg = train_cfg.get("checkpoint", {})
    ckpt_dir = f"checkpoints/{experiment_name}"
    if ckpt_dir:
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=ckpt_cfg.get("save_last", True),
        monitor=monitor_key,
        mode=ckpt_cfg.get("mode", "min"),
        save_top_k=ckpt_cfg.get("save_top_k", 3),
        filename=f"epoch{{epoch:02d}}_mAP{{{monitor_key}:.4f}}" + (f"-{experiment_name}" if experiment_name else ""),
        verbose=not cfg.get("silent", False),
        auto_insert_metric_name=False,
    )
    callbacks.append(ckpt_cb)

    es_cfg = train_cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=monitor_key,
                mode=es_cfg.get("mode", "min"),
                patience=es_cfg.get("patience", 5),
                min_delta=es_cfg.get("min_delta", 0.0),
                verbose=not cfg.get("silent", False),
            )
        )

    wandb_logger = WandbLogger(project=train_cfg.get("wandb_project", "GeoMetricLab-SupScene"), name=experiment_name)

    data_summary_cb = DatamoduleSummary(cfg.get("display_theme")) 
    model_framework_cb = ModelFrameworkSummary(cfg.get("display_theme"))
    model_summary_cb = CustomRichModelSummary(cfg.get("display_theme"))  
    progress_bar_cb = CustomRichProgressBar(cfg.get("display_theme"))     
    callbacks.append(data_summary_cb)
    callbacks.append(model_framework_cb)
    callbacks.append(model_summary_cb)
    callbacks.append(progress_bar_cb)

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 30),
        accelerator=train_cfg.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu"),
        devices=train_cfg.get("devices", "auto"),
        strategy=train_cfg.get("strategy", "auto"),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 0.0),
        precision=train_cfg.get("precision", 32),
        callbacks=callbacks,
        sync_batchnorm=train_cfg.get("sync_batchnorm", True),
        logger=wandb_logger,
        check_val_every_n_epoch=train_cfg.get("check_val_every_n_epoch", 1),
        fast_dev_run=cfg.get("dev", False),
        enable_model_summary=False, 
        enable_progress_bar=True, 
    )
    return trainer

def train(cfg):
    # Prepare data module
    dm = build_datamodule(cfg)

    # Model & loss
    model = build_model(cfg)
    criterion = build_criterion(cfg)

    # Lightning module
    lit_module = SupSceneFramework(
        model=model,
        criterion=criterion,
        optimizer_cfg=cfg["optimizer"],
        scheduler_cfg=cfg.get("scheduler", {}),
        use_proj=cfg.get("use_proj", False),
        use_bn=cfg.get("use_bn", False),
        verbose=not cfg.get("silent", False),
    )

    if cfg.get("compile", False):
        lit_module = torch.compile(lit_module)

    # Trainer
    monitor_key = cfg["trainer"].get("monitor", "val/mAP")
    experiment_name = cfg.get("experiment", "supscene-gl3d")
    trainer = build_trainer(cfg, monitor_key, experiment_name)

    # Optionally run validation once before training starts and save the
    # initial model state. Controlled by `trainer.validate_before_train` in
    # the config (boolean).
    if cfg["trainer"].get("validate_before_train", False):
        print_rank_0("[SupScene] Running validation before training...")
        try:
            val_out = trainer.validate(lit_module, datamodule=dm)
            print_rank_0(f"[SupScene] Pre-train validation results: {val_out}")
        except Exception as exc:
            print_rank_0(f"[SupScene] Pre-train validation failed: {exc}")

        # Save model weights (only the underlying model) to the experiment checkpoint dir
        ckpt_dir = f"checkpoints/{experiment_name}"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        before_path = Path(ckpt_dir) / "before_train_model.pth"
        try:
            torch.save(model.state_dict(), before_path)
            print_rank_0(f"[SupScene] Saved pre-training model to {before_path}")
        except Exception as exc:
            print_rank_0(f"[SupScene] Failed saving pre-training model: {exc}")

    print_rank_0("[SupScene] Starting training (new engine)...")
    trainer.fit(lit_module, datamodule=dm, ckpt_path=cfg["trainer"].get("resume_from"))

def main():
    from argparser import parse_args
    cfg = parse_args()
    seed_everything(cfg.get("seed", 42), workers=True)

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
    
    train(cfg)
    print_rank_0("Training completed.")


if __name__ == "__main__":
    main()
