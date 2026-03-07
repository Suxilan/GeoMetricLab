"""Instance Metric Learning LightningModule for PyTorch Lightning"""

from __future__ import annotations

from typing import Any, Dict, Optional

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.distributed as dist
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from pathlib import Path
from src.utils import load_features_h5, load_vlad_init_h5
from src.utils.metrics import compute_metrics, display_metrics
from src.utils.vis import aux_to_overlay_pil, tensor_to_rgb_pil
from src.models.projection import ProjectionHead
from src.models.geoencoder import GeoEncoder
from config.loss_hubs import CLASSIFICATION_LOSSES

class InstanceFramework(pl.LightningModule):
    """LightningModule for Instance Metric Learning.
    
    Handles instance batch format: (B, 3, H, W) -> (B, D) -> Loss
    """

    def __init__(
        self,
        model: GeoEncoder,
        criterion: nn.Module,
        optimizer_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        use_proj: bool = False,
        use_bn: bool = False,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion", "model"])

        self.model = model
        self.criterion = criterion
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg or {}
        self.verbose = verbose

        self.use_proj = use_proj
        self.use_bn = use_bn
        self.proj_head = None
        self.bn = None

        # If requested, create projection head / BN neck now so Lightning
        # can convert BatchNorm to SyncBatchNorm during trainer setup.
        if self.use_proj or self.use_bn:
            out_dim = getattr(self.model, "out_channels", None)
            if out_dim is None:
                raise ValueError("Model must expose `out_channels` when `use_proj=True` or `use_bn=True`")
        if self.use_proj:
            # create projection head (leave device placement to Lightning)
            self.proj_head = ProjectionHead(in_dim=out_dim)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model.
        
        Args:
            x: Input tensor of shape (B*N, C, H, W) or (B, N, C, H, W)
        
        Returns:
            Features of shape (B*N, D) or (B, N, D)
        """
        y = self.model(x)
        return y

    def _maybe_init_proj_head(self, in_dim) -> None:
        if self.proj_head is None:
            self.proj_head = ProjectionHead(in_dim=in_dim)

    def _maybe_init_bn(self, in_dim) -> None:
        if self.bn is None:
            self.bn = nn.BatchNorm1d(in_dim, affine=False)

    def _apply_projection(self, descriptors: torch.Tensor) -> torch.Tensor:
        if not self.use_proj:
            # maybe still have bn neck only
            if self.use_bn:
                self._maybe_init_bn(descriptors.shape[-1])
                out = self.bn(descriptors)
                return F.normalize(out, p=2, dim=-1)
            return descriptors

        # apply projection head first
        self._maybe_init_proj_head(descriptors.shape[-1])
        projected = self.proj_head(descriptors)

        # then optional BN neck
        if self.use_bn:
            self._maybe_init_bn(projected.shape[-1])
            projected = self.bn(projected)

        return F.normalize(projected, p=2, dim=-1)

    def _slice_aux_sample(self, aux_info: Dict[str, Any], i: int) -> Dict[str, torch.Tensor]:
        aux_cpu: Dict[str, torch.Tensor] = {}
        for k, v in aux_info.items():
            if not torch.is_tensor(v):
                continue
            if v.dim() >= 1:
                if v.size(0) == 1:
                    aux_cpu[k] = v[0].detach().cpu()
                elif v.size(0) > i:
                    aux_cpu[k] = v[i].detach().cpu()
                else:
                    aux_cpu[k] = v.detach().cpu()
            else:
                aux_cpu[k] = v.detach().cpu()
        return aux_cpu

    def _build_optimizer_param_groups(self, base_lr: float, base_wd: float):
        """Build minimal optimizer groups:
        - backbone group: `backbone_lr`
        - aggregator group: `aggregator_lr` (includes whitening layer)
        - default group: all other trainable params use global `lr`
        """
        backbone_lr = float(self.optimizer_cfg.get("backbone_lr", base_lr))
        aggregator_lr = float(self.optimizer_cfg.get("aggregator_lr", base_lr))

        named_params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        if len(named_params) == 0:
            raise RuntimeError("No trainable parameters found (all requires_grad=False).")

        backbone_params = []
        aggregator_params = []
        default_params = []

        for n, p in named_params:
            if n.startswith("model.backbone."):
                backbone_params.append(p)
            elif n.startswith("model.aggregator.") or n.startswith("model.whitening_layer."):
                aggregator_params.append(p)
            else:
                default_params.append(p)

        param_groups = []
        self._lr_group_names = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": base_wd})
            self._lr_group_names.append("backbone")
        if aggregator_params:
            param_groups.append({"params": aggregator_params, "lr": aggregator_lr, "weight_decay": base_wd})
            self._lr_group_names.append("aggregator")
        if default_params:
            param_groups.append({"params": default_params, "lr": base_lr, "weight_decay": base_wd})
            self._lr_group_names.append("default")

        if self.verbose:
            self.print(
                f"[optimizer] backbone_lr={backbone_lr}, aggregator_lr={aggregator_lr}, default_lr={base_lr}; "
                f"groups: backbone={len(backbone_params)}, aggregator+whitening={len(aggregator_params)}, default={len(default_params)}"
            )
        return param_groups
    
    def configure_optimizers(self):
        base_lr = float(self.optimizer_cfg.get("lr", 1e-4))
        base_wd = float(self.optimizer_cfg.get("weight_decay", 1e-4))
        param_groups = self._build_optimizer_param_groups(base_lr=base_lr, base_wd=base_wd)
        optimizer_name = self.optimizer_cfg.get("name", "").lower()

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=self.optimizer_cfg.get("momentum", 0.9))
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_cfg.get('name', '')}")
        
        warmup_start = float(self.scheduler_cfg.get("warmup_start_factor", 0.01))
        warmup_ratio = float(self.scheduler_cfg.get("warmup_ratio", 0.0))
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(total_steps * warmup_ratio)
        cosine_steps = max(total_steps - warmup_steps, 1)

        if warmup_steps > 0:
            scheduler1 = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=1e-7,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosine_steps,
                eta_min=1e-7,
            )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    # @torch.compiler.disable()
    def compute_loss(self, descriptors: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        compute label based metric learning loss.

        Args:
            descriptors: (B, D) feature descriptors
            labels: (B,) ground truth labels
        """

        loss, acc = self.criterion(descriptors, labels)
        return loss, acc

    def compute_total_loss(self, main_loss: torch.Tensor, aux_info: Optional[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine main task loss with optional auxiliary loss from aggregator."""
        aux_loss = torch.zeros((), device=main_loss.device, dtype=main_loss.dtype)
        if isinstance(aux_info, dict) and torch.is_tensor(aux_info.get("assign_map", None)):
            assign_map = aux_info["assign_map"]
            aggregator = getattr(self.model, "aggregator", None)
            if aggregator is not None and hasattr(aggregator, "compute_aux_loss"):
                extra = aggregator.compute_aux_loss(assign_map)
                if torch.is_tensor(extra):
                    aux_loss = extra.to(device=main_loss.device, dtype=main_loss.dtype)
        total_loss = main_loss + aux_loss
        return total_loss, aux_loss

    def init_whitening_from_h5(self, h5_path: Path) -> None:
        """
        Initialize whitening layer from precomputed features in H5 file.
        """
        # 1) 参数容器（在所有 rank 上都存在）
        weight = self.model.whitening_layer.weight
        bias = self.model.whitening_layer.bias
        device = weight.device  # 以参数所在 device 为准

        # 2) 只有 rank0 负责加载并计算
        if self.trainer.is_global_zero:
            self.print(f"[*] [Rank 0] Computing whitening from {h5_path}...")
            data = load_features_h5(h5_path)
            feats = data.get("features", None)

            if feats is not None:
                X = torch.from_numpy(feats).float().to(device)  # (N, D)
                self.model.init_whitening(X)        
            else:
                self.print("[!] No 'features' in H5. Will broadcast current (random) init.")

        # 3) DDP 同步：所有 rank 都必须执行
        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # 等 rank0 完成 copy_
            dist.broadcast(weight.data, src=0)  # ✅ 用 .data
            dist.broadcast(bias.data, src=0)
            dist.barrier()

        self.print(f"[*] [Rank {self.global_rank}] Whitening parameters synced.")

    def init_vlad_from_h5(self, h5_path: str) -> None:
        """Initialize *VLAD from cached H5 (`centers` + `descriptors`)."""
        agg = self.model.aggregator

        if self.trainer.is_global_zero:
            self.print(f"[*] Init *VLAD from cached H5: {h5_path}...")
            data = load_vlad_init_h5(Path(h5_path))
            centers = data["centers"]
            descriptors = data["descriptors"]
            self.print(f"   Loaded centers {centers.shape}, descriptors {descriptors.shape}.")
            agg.init_from_clusters(centers, descriptors)

        # --- 2. DDP: 同步参数给所有显卡 ---
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            if hasattr(agg, "centers"):
                dist.broadcast(agg.centers, src=0)
            dist.broadcast(agg.assign.weight, src=0)
            if getattr(agg.assign, "bias", None) is not None:
                dist.broadcast(agg.assign.bias, src=0)
            dist.barrier()

        self.print(f"[*] [Rank {self.global_rank}] *VLAD parameters synced.")
    
    def on_train_start(self):
        """
        Actions to perform at the start of training.
        """
        # If a whitening layer exists on the model, try to initialize it
        # from a precomputed feature cache: cache/features_resnet50_gem_115562.h5

        if getattr(self.model, "whitening", False):
            self.init_whitening_from_h5(
                h5_path=Path("features_resnet50_gem_115562.h5"),
            )

        # If using (*VLAD), optionally init from cached feature maps
        if getattr(self.model, "aggregator_name", "").lower() in ["netvlad", "supervlad"]:
            cache_h5 = self.optimizer_cfg.get("vlad_init_c_h5", None)
            if not cache_h5:
                raise ValueError("Missing optimizer.vlad_init_c_h5 for VLAD initialization.")
            self.init_vlad_from_h5(h5_path=Path(cache_h5))

    def _log_images(self, key: str, images: list[Image.Image | tuple[Image.Image, Image.Image]], caption_prefix: str = ""):
        if not images:
            return
        if isinstance(images[0], tuple):
            images_a = [img[0] for img in images]
            images_b = [img[1] for img in images]
            self._log_images(key, images_a, caption_prefix=caption_prefix)
            self._log_images(f"{key}_label", images_b, caption_prefix=caption_prefix)
            return
        experiment = getattr(self.logger, "experiment", None)
        
        if experiment and hasattr(experiment, "log"):

             try:
                wandb_imgs = [wandb.Image(img, caption=f"{caption_prefix}_{i}") for i, img in enumerate(images)]
                experiment.log({key: wandb_imgs, "epoch": self.current_epoch})
             except Exception as e:
                print(f"[Warn] Failed to log via self.logger: {e}")
        # ...

    def _create_vis_image(self, img_t: torch.Tensor, aux_t: object, key: str = "assign_map") -> Image.Image | tuple[Image.Image, Image.Image] | None:
        try:
            # 1. return to RGB
            pil_rgb = tensor_to_rgb_pil(img_t)
            
            # 2. generate heatmap (use your previous utility function)
            pil_overlay = aux_to_overlay_pil(aux_t, pil_rgb.size, key=key, cmap="jet") # viridis, plasma, inferno, magma, cividis, jet
            
            if pil_overlay is None:
                return None
                
            # 3. fusion
            if isinstance(pil_overlay, tuple):
                blended = []
                for overlay in pil_overlay:
                    if overlay.mode != 'RGB':
                        overlay = overlay.convert('RGB')
                    blended.append(Image.blend(pil_rgb, overlay, alpha=0.45))
                return blended[0], blended[1]
            else:
                if pil_overlay.mode != 'RGB':
                    pil_overlay = pil_overlay.convert('RGB')
                return Image.blend(pil_rgb, pil_overlay, alpha=0.45)
        except Exception as e:
            self.print(f"[Warn] Vis failed: {e}")
            return None

    ########################################################
    ################ Training loop starts here #############
    ########################################################
    def on_train_epoch_start(self):
        """
        Actions to perform at the start of each training epoch.
        """
        if self.global_rank == 0:
            self.train_aux_samples = []
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for subgraph batch.
        
        Args:
            batch: Dict with keys:
                - images: (B, 3, H, W) or None
                - labels: (B,) or None
            batch_idx: Batch index
        
        Returns:
            Loss tensor
        """
        images, labels = batch
        model_output = self(images)  # (B*N, D)

        # some models may return tuple outputs(x, auxiliary_data)
        aux_info = None
        if isinstance(model_output, (tuple, list)):
            descriptors = model_output[0]
            if len(model_output) > 1:
                aux_info = model_output[1]
        else:
            descriptors = model_output

        if isinstance(aux_info, dict) and "p" in aux_info and torch.is_tensor(aux_info["p"]):
            p_vals = aux_info["p"].reshape(-1).float()
            p_mean = p_vals.mean()
            p_std = p_vals.std(unbiased=False)
            p_min_v = p_vals.min()
            p_max_v = p_vals.max()
            self.log("PStats/mean", p_mean, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("PStats/std", p_std, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("PStats/min", p_min_v, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("PStats/max", p_max_v, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)


        VIS_LIMIT = 4
        if self.global_rank == 0 and aux_info is not None:
            self.train_aux_samples = [] 
            num_to_keep = min(VIS_LIMIT, images.size(0))
            for i in range(num_to_keep):
                img_cpu = images[i].detach().cpu()
                aux_cpu = None
                if isinstance(aux_info, dict):
                    aux_cpu = self._slice_aux_sample(aux_info, i)
                elif torch.is_tensor(aux_info):
                    aux_cpu = aux_info[i].detach().cpu()
                
                if aux_cpu is not None:
                    self.train_aux_samples.append((img_cpu, aux_cpu))

        descriptors = self._apply_projection(descriptors)

        # Compute loss
        # For some classification-style losses we MUST NOT offset labels across ranks
        loss_name = getattr(self.criterion, "loss_fn_name", None)
        if loss_name is None:
            # fallback: do not alter labels
            pass
        else:
            # if base loss belongs to CLASSIFICATION_LOSSES, skip rank offset
            if loss_name not in CLASSIFICATION_LOSSES:
                rank = self.global_rank
                labels = labels + rank * 100000  # for distributed training

        main_loss, acc = self.compute_loss(descriptors, labels)
        loss, aux_loss = self.compute_total_loss(main_loss, aux_info)

        # Logging
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0), sync_dist=True)
        self.log("train/loss_main", main_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=images.size(0), sync_dist=True)
        self.log("train/loss_aux", aux_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=images.size(0), sync_dist=True)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0), sync_dist=True)
        if isinstance(aux_info, dict) and "discard_map" in aux_info and torch.is_tensor(aux_info["discard_map"]):
            discard_map = aux_info["discard_map"]
            discard_rate = discard_map.mean()
            self.log("train/discard_rate", discard_rate, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.size(0), sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        """
        Actions to perform at the end of each training epoch.
        """
        if self.global_rank != 0 or not getattr(self, "train_aux_samples", None):
            return

        # 2. process images
        processed_assign = []
        processed_discard = []
        for img_t, aux_t in self.train_aux_samples:
            pil_assign = self._create_vis_image(img_t, aux_t, key="assign_map")
            if pil_assign:
                processed_assign.append(pil_assign)
            pil_discard = self._create_vis_image(img_t, aux_t, key="discard_map")
            if pil_discard:
                processed_discard.append(pil_discard)

        # 3. log images
        self._log_images("train_aux/assign_vis", processed_assign, caption_prefix=f"ep{self.current_epoch}")
        self._log_images("train_aux/discard_vis", processed_discard, caption_prefix=f"ep{self.current_epoch}")

        # 4. clear samples
        self.train_aux_samples = []
    
    ########################################################
    ################ Validation loop starts here ###########
    ########################################################
    def on_validation_epoch_start(self):
        """
        Actions to perform at the start of each validation epoch.
        """
        # we init an empty dictionary to store the descriptors for each dataloader
        self.validation_step_outputs = {}
        if self.global_rank == 0:
            self.val_aux_samples = []

    def validation_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Validation step for feature extraction.
        
            Args:
            batch: Dict with keys:
                - image: (B, C, H, W) - single image for evaluation
                - index: (B,) - sample indices
            batch_idx: Batch index
        
        Returns:
            Dict with keys:
                - features: (B, D) - extracted features
                - indices: (B,) - sample indices    """
        images, indices = batch
        model_output = self(images)  # (B, D), ()

        # some models may return tuple outputs(x, auxiliary_data)
        aux_info = None
        if isinstance(model_output, (tuple, list)):
            descriptors = model_output[0]
            if len(model_output) > 1:
                aux_info = model_output[1]
        else:
            descriptors = model_output

        VIS_LIMIT = 4
        if self.global_rank == 0 and aux_info is not None:
            self.train_aux_samples = [] 
            num_to_keep = min(VIS_LIMIT, images.size(0))
            for i in range(num_to_keep):
                img_cpu = images[i].detach().cpu()
                aux_cpu = None
                if isinstance(aux_info, dict):
                    aux_cpu = self._slice_aux_sample(aux_info, i)
                elif torch.is_tensor(aux_info):
                    aux_cpu = aux_info[i].detach().cpu()
                
                if aux_cpu is not None:
                    self.train_aux_samples.append((img_cpu, aux_cpu))
                  
        if dataloader_idx not in self.validation_step_outputs:
            # initialize the list of descriptors for this dataloader
            self.validation_step_outputs[dataloader_idx] = {'descriptors': [], 'indices': []}
    
        self.validation_step_outputs[dataloader_idx]['descriptors'].append(descriptors.detach().cpu())
        self.validation_step_outputs[dataloader_idx]['indices'].append(indices.detach().cpu())

    def on_validation_epoch_end(self):
        """
        End of validation: Gather, Sort, Compute Metrics.
        """
        dm = self.trainer.datamodule
        
        # 遍历每个验证集 (dataloader)
        for dataloader_idx, outputs in self.validation_step_outputs.items():
            
            dataset = dm.val_datasets[dataloader_idx]
            dataset_name = dataset.dataset_name

            if self.trainer.fast_dev_run:
                if dataloader_idx == 0:
                    self.print("\n[Fast Dev Run] Skipping expensive recall computation.\n")
                continue
            
            if self.trainer.sanity_checking:
                if dataloader_idx == 0:
                    self.print("\n[Sanity Check] Skipping expensive recall computation.\n")
                continue
            
            local_descriptors = torch.cat(outputs['descriptors'], dim=0)  # (num_local, D)
            local_indices = torch.cat(outputs['indices'], dim=0)          # (num_local,)

            sorted_descriptors, sorted_indices = self._gather_distributed_data(
                local_descriptors, local_indices
            )
            metrics = {}
            
            if self.trainer.is_global_zero:
                # 跳过 Fast Dev Run (冒烟测试)
                if "gl3d" in dataset_name.lower():
                    k_values = [1, 25, 50, 100]
                    filter_keys = ["CMC"]
                    sorted_descriptors = torch.cat([sorted_descriptors, sorted_descriptors], dim=0)
                    exclude_self = True
                elif "u1652" in dataset_name.lower():
                    k_values = [1, 5, 10]
                    filter_keys = ["R"]
                    exclude_self = False
                else:
                    k_values = [1, 10, 20]
                    filter_keys = []

                raw_metrics = compute_metrics(
                    descriptors=sorted_descriptors, 
                    num_references=dataset.num_references,
                    num_queries=dataset.num_queries,
                    ground_truth=dataset.ground_truth,
                    k_values=k_values, 
                    metric="cosine", 
                    exclude_self=exclude_self,
                )

                metrics = {
                    k: v for k, v in raw_metrics.items() 
                    if not any(fk in k for fk in filter_keys) and "num_queries_total" not in k and "num_valid_queries" not in k
                }

                if self.verbose:
                    display_metrics(metrics, title=dataset_name)

            if self.trainer.world_size > 1:
                metrics_container = [metrics] 
                dist.broadcast_object_list(metrics_container, src=0)
                metrics = metrics_container[0]

            if metrics:
                log_metrics = {f"{dataset_name}-val/{k}": v for k, v in metrics.items()}
                self.log_dict(log_metrics, prog_bar=False, logger=True, sync_dist=False)

        self.validation_step_outputs.clear()

        ##############################
        ### Visualization aux maps ###
        ##############################
        if self.global_rank != 0 or not getattr(self, "val_aux_samples", None):
            return

        processed_assign = []
        processed_discard = []
        for img_t, aux_t in self.val_aux_samples:
            pil_assign = self._create_vis_image(img_t, aux_t, key="assign_map")
            if pil_assign:
                processed_assign.append(pil_assign)
            pil_discard = self._create_vis_image(img_t, aux_t, key="discard_map")
            if pil_discard:
                processed_discard.append(pil_discard)

        self._log_images("val_aux/assign_vis", processed_assign, caption_prefix=f"ep{self.current_epoch}")
        self._log_images("val_aux/discard_vis", processed_discard, caption_prefix=f"ep{self.current_epoch}")

        self.val_aux_samples = []

    def _gather_distributed_data(self, local_descriptors, local_indices):
        """Gather and deduplicate descriptors/indices across processes.

        Important: this runs at validation epoch end and can involve millions of
        descriptors. Using Lightning's `all_gather` will typically move tensors
        to CUDA (NCCL) and can easily OOM. We therefore gather on CPU and only
        materialize the full set on rank 0.

        Args:
            local_descriptors: CPU tensor (N, D)
            local_indices: CPU tensor (N,)

        Returns:
            (rank0) sorted_descriptors, sorted_indices on CPU
            (non-rank0) empty CPU tensors (not used for metric computation)
        """

        # Ensure CPU tensors (safe for gather_object serialization)
        if local_descriptors.is_cuda:
            local_descriptors = local_descriptors.detach().cpu()
        if local_indices.is_cuda:
            local_indices = local_indices.detach().cpu()

        world_size = getattr(self.trainer, "world_size", 1)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()

            # Prefer gather-to-rank0 to avoid materializing full tensors on every rank.
            if hasattr(dist, "gather_object"):
                if rank == 0:
                    gathered = [None for _ in range(world_size)]
                    dist.gather_object((local_descriptors, local_indices), object_gather_list=gathered, dst=0)
                else:
                    dist.gather_object((local_descriptors, local_indices), object_gather_list=None, dst=0)
                    return (
                        torch.empty((0, local_descriptors.shape[-1]), dtype=local_descriptors.dtype),
                        torch.empty((0,), dtype=local_indices.dtype),
                    )
            else:
                # Fallback for older torch: everyone receives the full object list.
                gathered = [None for _ in range(world_size)]
                dist.all_gather_object(gathered, (local_descriptors, local_indices))
                if rank != 0:
                    return (
                        torch.empty((0, local_descriptors.shape[-1]), dtype=local_descriptors.dtype),
                        torch.empty((0,), dtype=local_indices.dtype),
                    )

            desc_list = []
            idx_list = []
            for item in gathered:
                if item is None:
                    continue
                d, i = item
                if d is None or i is None:
                    continue
                desc_list.append(d)
                idx_list.append(i)

            if len(desc_list) == 0:
                return (
                    torch.empty((0, 0), dtype=local_descriptors.dtype),
                    torch.empty((0,), dtype=local_indices.dtype),
                )

            all_descriptors = torch.cat(desc_list, dim=0)
            all_indices = torch.cat(idx_list, dim=0)
        else:
            all_descriptors = local_descriptors
            all_indices = local_indices

        # Rank0 / single-process: sort + dedup on CPU
        if all_indices.numel() == 0:
            return all_descriptors, all_indices

        order = torch.argsort(all_indices)
        sorted_descriptors = all_descriptors[order]
        sorted_indices = all_indices[order]

        # Only allocate a filtered copy if duplicates actually exist
        if sorted_indices.numel() > 1 and torch.any(sorted_indices[1:] == sorted_indices[:-1]):
            mask = torch.ones_like(sorted_indices, dtype=torch.bool)
            mask[1:] = sorted_indices[1:] != sorted_indices[:-1]
            sorted_descriptors = sorted_descriptors[mask]
            sorted_indices = sorted_indices[mask]

        return sorted_descriptors, sorted_indices