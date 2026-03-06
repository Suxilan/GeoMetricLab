"""Training engine: UCA on University1652 (feature-based).

This is a lightweight PyTorch Lightning engine that wires together:
- UCADataModule (feature cache)
- AttnAgg (sequence reranker)
- UCAFramework (training + validation)

Usage:
	python engine/train_uca_u1652_engine.py --config config/train_uca_u1652/default.yaml

Notes:
- Defaults are intentionally minimal. Provide a YAML config for real runs.
- Validation metrics are computed on a single GPU/CPU in UCAFramework.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch

# project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import print_rank_0
from src.pipeline.uca.uca_datamodule import UCADataModule
from src.pipeline.uca.uca_framework import UCAFramework
from src.models.reranker.attnagg import AttnAgg
from src.utils.callbacks import CustomRichModelSummary, CustomRichProgressBar


def build_criterion(cfg: Dict[str, Any]):
	loss_cfg = cfg.get("loss", {}) or {}
	module_name = loss_cfg.get("module")
	class_name = loss_cfg.get("class")
	params = loss_cfg.get("params", {}) or {}

	if not module_name or not class_name:
		raise ValueError("Loss configuration must include 'module' and 'class' entries.")

	module = importlib.import_module(module_name)
	loss_cls = getattr(module, class_name)
	return loss_cls(**params)

def build_datamodule(cfg: Dict[str, Any]) -> UCADataModule:
	data_cfg = cfg.get("data", {}) or {}

	train_cache_root = data_cfg.get("train_cache_root") or data_cfg.get("train_datapath") or "cache/u1652"
	val_cache_root = data_cfg.get("val_cache_root") or data_cfg.get("val_datapath") or train_cache_root

	train_batch_size = int(data_cfg.get("train_batch_size") or data_cfg.get("batch_size") or 64)
	val_batch_size = int(data_cfg.get("val_batch_size") or 32)
	num_workers = int(data_cfg.get("num_workers") or 4)
	max_k = int(data_cfg.get("max_k") or 100)

	val_tasks = data_cfg.get("val_tasks") or data_cfg.get("val_set_names") or [
		"u1652_drone2satellite",
		"u1652_satellite2drone",
	]

	return UCADataModule(
		train_cache_root=str(train_cache_root),
		val_cache_root=str(val_cache_root),
		train_batch_size=train_batch_size,
		val_batch_size=val_batch_size,
		num_workers=num_workers,
		max_k=max_k,
		val_tasks=val_tasks,
	)


def build_model(cfg: Dict[str, Any]) -> AttnAgg:
	# For UCA, cfg['model'] is expected to be AttnAgg kwargs.
	model_cfg = cfg.get("model", {})
	if not isinstance(model_cfg, dict):
		model_cfg = {}

	embed_dim = int(model_cfg.get("embed_dim", 512))
	num_heads = int(model_cfg.get("num_heads", 8))
	num_layers = int(model_cfg.get("num_layers", 2))
	mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
	dropout = float(model_cfg.get("dropout", 0.0))
	rope_base = float(model_cfg.get("rope_base", 10000.0))

	return AttnAgg(
		embed_dim=embed_dim,
		num_heads=num_heads,
		num_layers=num_layers,
		mlp_ratio=mlp_ratio,
		dropout=dropout,
		rope_base=rope_base,
	)


def build_trainer(cfg: Dict[str, Any], *, experiment_name: str) -> pl.Trainer:
	train_cfg = cfg.get("trainer", {}) or {}

	callbacks = [LearningRateMonitor(logging_interval="step")]

	monitor_key = str(train_cfg.get("monitor", "u1652_drone2satellite-val/mAP"))
	ckpt_cfg = train_cfg.get("checkpoint", {}) or {}
	ckpt_dir = Path("checkpoints") / experiment_name
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	callbacks.append(
		ModelCheckpoint(
			dirpath=str(ckpt_dir),
			save_last=bool(ckpt_cfg.get("save_last", True)),
			monitor=monitor_key,
			mode=str(ckpt_cfg.get("mode", "max")),
			save_top_k=int(ckpt_cfg.get("save_top_k", 3)),
			filename=f"epoch{{epoch:02d}}_{{{monitor_key}:.4f}}-{experiment_name}",
			verbose=not cfg.get("silent", False),
			auto_insert_metric_name=False,
		)
	)

	model_summary_cb = CustomRichModelSummary(cfg.get("display_theme"))  
	progress_bar_cb = CustomRichProgressBar(cfg.get("display_theme"))     
	callbacks.append(model_summary_cb)
	callbacks.append(progress_bar_cb)

	wandb_logger = WandbLogger(project=train_cfg.get("wandb_project", "GeoMetricLab-UCA"), name=experiment_name)

	trainer = pl.Trainer(
		max_epochs=int(train_cfg.get("max_epochs", 50)),
		accelerator=train_cfg.get("accelerator", "gpu" if torch.cuda.is_available() else "cpu"),
		devices=train_cfg.get("devices", "auto"),
		strategy=train_cfg.get("strategy", "auto"),
		log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
		gradient_clip_val=float(train_cfg.get("gradient_clip_val", 0.0)),
		precision=train_cfg.get("precision", 32),
		sync_batchnorm=bool(train_cfg.get("sync_batchnorm", False)),
		check_val_every_n_epoch=int(train_cfg.get("check_val_every_n_epoch", 1)),
		fast_dev_run=cfg.get("dev", False),
		callbacks=callbacks,
		logger=wandb_logger,
		enable_model_summary=False,
		enable_progress_bar=True,
	)
	return trainer


def train(cfg: Dict[str, Any]) -> None:
	dm = build_datamodule(cfg)
	model = build_model(cfg)
	criterion = build_criterion(cfg)

	framework_cfg = cfg.get("framework", {}) or {}
	lit_module = UCAFramework(
		model=model,
		metric_criterion=criterion,
		optimizer_cfg=cfg.get("optimizer", {}) or {},
		scheduler_cfg=cfg.get("scheduler", {}) or {},
		topk_drop_prob=float(framework_cfg.get("topk_drop_prob", cfg.get("topk_drop_prob", 0.3))),
		metric_weight=float(framework_cfg.get("metric_weight", cfg.get("metric_weight", 1.0))),
		aux_weight=float(framework_cfg.get("aux_weight", cfg.get("aux_weight", 1.0))),
		verbose=not cfg.get("silent", False),
	)

	if cfg.get("compile", False):
		lit_module = torch.compile(lit_module)

	experiment_name = str(cfg.get("experiment", "uca-u1652"))
	trainer = build_trainer(cfg, experiment_name=experiment_name)

	print_rank_0("[UCA-U1652] Starting training...")
	trainer.fit(lit_module, datamodule=dm, ckpt_path=(cfg.get("trainer", {}) or {}).get("resume_from"))
	print_rank_0("Training completed.")


def main() -> None:
	from argparser import parse_args

	cfg = parse_args()
	seed_everything(int(cfg.get("seed", 42)), workers=True)

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


if __name__ == "__main__":
	main()