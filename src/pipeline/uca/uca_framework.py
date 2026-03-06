"""UCA training framework."""

from __future__ import annotations
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.utils.metrics import compute_metrics, display_metrics


class UCAFramework(pl.LightningModule):
	def __init__(
		self,
		model: nn.Module,
		metric_criterion: nn.Module,
		*,
		optimizer_cfg: Dict[str, Any],
		scheduler_cfg: Optional[Dict[str, Any]] = None,
		topk_drop_prob: float = 0.3,
		metric_weight: float = 1.0,
		aux_weight: float = 1.0,
		verbose: bool = True,
	) -> None:
		super().__init__()
		self.save_hyperparameters(ignore=["model", "metric_criterion"])
		self.model = model
		self.metric_criterion = metric_criterion
		self.optimizer_cfg = optimizer_cfg
		self.scheduler_cfg = scheduler_cfg or {}
		self.topk_drop_prob = float(topk_drop_prob)
		self.metric_weight = float(metric_weight)
		self.aux_weight = float(aux_weight)
		self.verbose = bool(verbose)
		self.aux_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

	def configure_optimizers(self):
		name = str(self.optimizer_cfg.get("name", "adamw")).lower()
		lr = float(self.optimizer_cfg.get("lr", 1e-4))
		wd = float(self.optimizer_cfg.get("weight_decay", 1e-4))
		aux_lr = float(self.optimizer_cfg.get("aux_lr", lr))
		aux_wd = float(self.optimizer_cfg.get("aux_weight_decay", wd))

		aux_params = []
		if hasattr(self.model, "aux_classifier") and isinstance(getattr(self.model, "aux_classifier"), nn.Module):
			aux_params = list(self.model.aux_classifier.parameters())
		aux_param_ids = {id(p) for p in aux_params}
		main_params = [p for p in self.parameters() if p.requires_grad and id(p) not in aux_param_ids]
		param_groups = [{"params": main_params, "lr": lr, "weight_decay": wd}]
		if aux_params:
			param_groups.append({"params": aux_params, "lr": aux_lr, "weight_decay": aux_wd})

		if name == "adamw":
			opt = torch.optim.AdamW(param_groups)
		elif name == "sgd":
			opt = torch.optim.SGD(param_groups, momentum=float(self.optimizer_cfg.get("momentum", 0.9)))
		else:
			raise ValueError(f"Unsupported optimizer: {name}")

		max_epochs = int(getattr(self.trainer, "max_epochs", 0) or 0)
		warmup_ratio = float(self.scheduler_cfg.get("warmup_ratio", 0.0))
		if max_epochs <= 0 or warmup_ratio <= 0.0:
			return opt

		warmup_epochs = int(max_epochs * warmup_ratio)
		warmup_start = float(self.scheduler_cfg.get("warmup_start_factor", 0.01))
		s1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=warmup_start, end_factor=1.0, total_iters=warmup_epochs)
		s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-7)
		sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_epochs])
		return [opt], [{"scheduler": sched, "interval": "epoch"}]

	def _augment_key_padding_mask(self, base_mask: torch.Tensor) -> torch.Tensor:
		# base_mask: (B, L) bool, True means masked
		if self.topk_drop_prob <= 0.0:
			return base_mask
		b, l = base_mask.shape
		if l <= 1:
			return base_mask
		mask = base_mask.clone()
		valid = ~mask
		valid[:, 0] = False
		drop = (torch.rand((b, l), device=mask.device) < self.topk_drop_prob) & valid
		mask |= drop
		return mask

	def _compute_aux_loss(self, aux_logits: torch.Tensor, labels: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
		# aux_logits: (B, K), labels: (B, K) bool/0-1, key_padding_mask: (B, L=K+1)
		if aux_logits.ndim != 2:
			raise ValueError(f"Expected aux_logits (B,K), got {tuple(aux_logits.shape)}")
		if labels.shape != aux_logits.shape:
			raise ValueError(f"labels shape {tuple(labels.shape)} must match aux_logits {tuple(aux_logits.shape)}")
		target = labels.to(device=aux_logits.device, dtype=aux_logits.dtype)
		valid = ~key_padding_mask[:, 1:]
		loss = self.aux_loss_fn(aux_logits, target)
		loss = loss[valid]
		return loss.mean() if loss.numel() > 0 else aux_logits.new_tensor(0.0)

	def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
		drone_seq = batch["drone_seq"].to(dtype=torch.float32)
		sat_seq = batch["sat_seq"].to(dtype=torch.float32)
		drone_labels = batch["drone_labels"].to(device=drone_seq.device)
		sat_labels = batch["sat_labels"].to(device=sat_seq.device)
		labels = batch.get("label", None)

		if labels is None:
			raise KeyError("UCA batch must contain `label` for metric loss")
		labels = labels.to(device=drone_seq.device, dtype=torch.long)

		b, l, _ = drone_seq.shape
		k = int(l - 1)
		# k_min = max(1, k // 2)
		k_min = max(1, k)
		k_eff = int(torch.randint(low=k_min, high=k + 1, size=(1,), device=drone_seq.device).item())
		l_eff = 1 + k_eff

		drone_seq = drone_seq[:, :l_eff, :]
		sat_seq = sat_seq[:, :l_eff, :]
		drone_labels = drone_labels[:, :k_eff]
		sat_labels = sat_labels[:, :k_eff]

		base_mask = torch.zeros((b, l_eff), dtype=torch.bool, device=drone_seq.device)
		drone_mask = self._augment_key_padding_mask(base_mask)
		sat_mask = self._augment_key_padding_mask(base_mask)

		q_drone_out, aux_drone_logits = self.model(drone_seq, key_padding_mask=drone_mask)
		q_sat_out, aux_sat_logits = self.model(sat_seq, key_padding_mask=sat_mask)

		# Metric loss (instance-style): concatenate both branches so other scenes
		# in the batch naturally serve as negatives.
		embeddings = torch.cat([q_drone_out, q_sat_out], dim=0)
		metric_labels = torch.cat([labels, labels], dim=0)
		loss_metric, acc_metric = self.metric_criterion(embeddings, metric_labels)
		loss_aux_d = self._compute_aux_loss(aux_drone_logits, drone_labels, drone_mask)
		loss_aux_s = self._compute_aux_loss(aux_sat_logits, sat_labels, sat_mask)
		loss_aux = 0.5 * (loss_aux_d + loss_aux_s)
		loss = (self.metric_weight * loss_metric) + (self.aux_weight * loss_aux)

		self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=b)
		self.log("train/loss_metric", loss_metric, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=b)
		self.log("train/acc_metric", acc_metric, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=b)
		self.log("train/loss_aux", loss_aux, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=b)
		return loss
	
	def on_train_epoch_end(self):
		"""
		Actions to perform at the end of each training epoch.
		"""
		pass

	def on_validation_epoch_start(self):
		"""
		Actions to perform at the start of each validation epoch.
		"""
		# store per-dataloader descriptors and indices
		self.validation_step_outputs: Dict[int, Dict[str, list[torch.Tensor]]] = {}
		
	def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		# val dataloader returns: (seq, index)
		seq, indices = batch
		seq = seq.to(dtype=torch.float32)
		q, _ = self.model(seq, key_padding_mask=None)
		descriptors = q
		if dataloader_idx not in self.validation_step_outputs:
			self.validation_step_outputs[dataloader_idx] = {"descriptors": [], "indices": []}
		self.validation_step_outputs[dataloader_idx]["descriptors"].append(descriptors.detach().cpu())
		self.validation_step_outputs[dataloader_idx]["indices"].append(indices.detach().cpu())
		return
	
	def on_validation_epoch_end(self):
		"""
		End of validation: Gather, Sort, Compute Metrics.
		"""
		dm = self.trainer.datamodule
		task_sums: Dict[str, Dict[str, float]] = {}
		task_counts: Dict[str, int] = {}
		
		# 遍历每个验证集 (dataloader)
		for dataloader_idx, outputs in self.validation_step_outputs.items():
			
			dataset = dm.val_datasets[dataloader_idx]
			dataset_name = dataset.dataset_name
			task_name = getattr(dataset, "task_name", dataset_name)

			if self.trainer.fast_dev_run:
				if dataloader_idx == 0:
					self.print("\n[Fast Dev Run] Skipping expensive recall computation.\n")
				continue
			
			if self.trainer.sanity_checking:
				if dataloader_idx == 0:
					self.print("\n[Sanity Check] Skipping expensive recall computation.\n")
				continue
			
			local_descriptors = torch.cat(outputs["descriptors"], dim=0)  # (N, D)
			local_indices = torch.cat(outputs["indices"], dim=0)          # (N,)
			sorted_descriptors = local_descriptors

			k_values = [1, 5, 10]
			filter_keys = ["R"] if "u1652" in str(dataset_name).lower() else []
			raw_metrics = compute_metrics(
				descriptors=sorted_descriptors,
				num_references=dataset.num_references,
				num_queries=dataset.num_queries,
				ground_truth=dataset.ground_truth,
				k_values=k_values,
				metric="cosine",
				exclude_self=False,
			)
			metrics = {
				k: v for k, v in raw_metrics.items()
				if not any(fk in k for fk in filter_keys) and "num_queries_total" not in k and "num_valid_queries" not in k
			}

			if self.verbose:
				display_metrics(metrics, title=dataset_name)
			if metrics:
				# 1) per-feature-file metrics (unique prefix, won't overwrite)
				log_metrics = {f"{dataset_name}-val/{k}": v for k, v in metrics.items()}
				self.log_dict(log_metrics, prog_bar=False, logger=True, sync_dist=False)

				# 2) per-task aggregated metrics (stable prefix for monitoring)
				if task_name not in task_sums:
					task_sums[task_name] = {mk: 0.0 for mk in metrics.keys()}
					task_counts[task_name] = 0
				for mk, mv in metrics.items():
					task_sums[task_name][mk] += float(mv)
				task_counts[task_name] += 1

		# log per-task averages (single card)
		for task_name, sums in task_sums.items():
			cnt = max(1, int(task_counts.get(task_name, 1)))
			avg_metrics = {f"{task_name}-val/{k}": (v / cnt) for k, v in sums.items()}
			self.log_dict(avg_metrics, prog_bar=False, logger=True, sync_dist=False)

		self.validation_step_outputs.clear()