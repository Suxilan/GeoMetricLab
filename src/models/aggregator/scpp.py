import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any, Dict, Optional

from .base import AggregatorBase

class SCPP(AggregatorBase):
	"""Structural Context Probe Pooling aggregator.
	"""
	def __init__(
		self,
		in_channels: int,
		p_s: float = 1.3,
		p_a: float = 4.6,
		eps: float = 1e-6
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = in_channels * 2
		self.p_s = float(p_s)
		self.p_a = float(p_a)
		self.eps = float(eps)

		self.dwconv = nn.Conv2d(
			in_channels, in_channels,
			kernel_size=3, padding=1,
			groups=in_channels, bias=True
		)

		self.confidence_proj_low = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
		self.confidence_proj_high = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

	def _power_pool(self, x: torch.Tensor, conf_map: torch.Tensor, p: float) -> torch.Tensor:
		conf_norm = conf_map / conf_map.mean(dim=(2, 3), keepdim=True).clamp_min(self.eps)

		x_abs = x.abs().clamp_min(self.eps)
		x_pow = x_abs.pow(p)

		weighted_sum = (conf_norm * x_pow).sum(dim=(2, 3))                  # (B, C)
		weight_sum = conf_norm.sum(dim=(2, 3)).clamp_min(self.eps)          # (B, 1)
		pooled = (weighted_sum / weight_sum).pow(1.0 / p)                   # (B, C)

		return pooled, conf_norm

	def forward(
		self,
		x: torch.Tensor,
	) -> Tuple[torch.Tensor, Dict[str, Any]]:
		B, C, H, W = x.shape

		h = x + self.dwconv(x)

		conf_map_low = torch.sigmoid(self.confidence_proj_low(h))            # (B, 1, H, W)
		conf_map_high = torch.sigmoid(self.confidence_proj_high(h))          # (B, 1, H, W)

		pooled_support, conf_norm_low = self._power_pool(x, conf_map_low, self.p_s)
		pooled_anchor, conf_norm_high = self._power_pool(x, conf_map_high, self.p_a)

		v_global = torch.cat([F.normalize(pooled_support, p=2, dim=1),
							  F.normalize(pooled_anchor, p=2, dim=1)], dim=1)

		return v_global, {
			"assign_map": conf_map_low
		}

	# def __init__(self, in_channels: int, num_clusters: int = None, p: float = 3.0, eps: float = 1e-6):
	# 	super().__init__()
	# 	self.out_channels = in_channels
	# 	self.in_channels = in_channels
	# 	self.num_clusters = num_clusters
	# 	self.p = float(p)
	# 	self.eps = float(eps)

	# 	self.dwconv = nn.Conv2d(
	# 		in_channels, in_channels,
	# 		kernel_size=3, padding=1,
	# 		groups=in_channels, bias=True
	# 	)
	# 	self.confidence_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

	# def forward(
	# 	self,
	# 	x: torch.Tensor,
	# 	attn: torch.Tensor = None,
	# 	cls: torch.Tensor = None
	# ) -> Tuple[torch.Tensor, Dict[str, Any]]:
	# 	B, C, H, W = x.shape

	# 	h = x + self.dwconv(x)
	# 	conf_logits = self.confidence_proj(h)                      # (B, 1, H, W)
	# 	conf_map = torch.sigmoid(conf_logits)                      # (B, 1, H, W)

	# 	# 归一化到均值约为 1，避免整体缩放不稳定
	# 	conf_norm = conf_map / conf_map.mean(dim=(2, 3), keepdim=True).clamp_min(self.eps)

	# 	x_abs = x.abs().clamp_min(self.eps)
	# 	x_pow = x_abs.pow(self.p)

	# 	weighted_sum = (conf_norm * x_pow).sum(dim=(2, 3))        # (B, C)
	# 	weight_sum = conf_norm.sum(dim=(2, 3)).clamp_min(self.eps)  # (B, 1)
	# 	v = (weighted_sum / weight_sum).pow(1.0 / self.p)    # (B, C)

	# 	return v, {"assign_map": conf_norm.detach()}