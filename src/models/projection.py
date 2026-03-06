from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """Simple 2-layer projection head (SimCLR style).

    Applies: Linear -> ReLU -> Linear. Shapes are preserved for 2D inputs; for
    higher-rank tensors we flatten the last dimension, project, then reshape back.
    """

    def __init__(self, in_dim: int, hidden_dim=2048, out_dim=256):
        super().__init__()
        hidden = hidden_dim
        out = out_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out, bias=False),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        if isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., D_in).

        Returns:
            z: Output tensor of shape (..., D_out).
        """
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)

        return z

class RetrievalHead(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512):
        super().__init__()
        # 1. 降维 (Projection)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        
        # 2. BatchNorm (非常关键！BNNeck 的核心)
        # 它可以把特征拉回到超球面上，有利于 Cosine 相似度计算
        self.bn = nn.BatchNorm1d(out_dim)
        
        # 3. 初始化 (如果是 Linear，通常用 Kaiming 或者 Orthogonal)
        nn.init.orthogonal_(self.proj.weight)
        
    def forward(self, x):
        feat = self.proj(x)
        feat = self.bn(feat)
        return feat

# Code adapted from MocoV3
# https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
class MoCoProjection(nn.Module):
    """
    MoCo/SimCLR style projection head: multi-layer MLP + optional last layer BN (no affine)
    """

    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        hidden_dim: int = 2048,
        nlayers: int = 3, 
        last_bn: bool = False
    ):
        super().__init__()
        self.head = self._build_mlp(nlayers, in_dim, hidden_dim, out_dim, last_bn=last_bn)
        
    def _build_mlp(self, nlayers, in_dim, hidden_dim, out_dim, last_bn=True):
        """构建MLP"""
        mlp = []
        for l in range(nlayers):
            dim1 = in_dim if l == 0 else hidden_dim
            dim2 = out_dim if l == nlayers - 1 else hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < nlayers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # SimCLR uses a BatchNorm without affine transformation
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
        
    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D_in]   
            node_mask: [B, N] 
        Returns:
            z: [B, N, D_out] 
        """
        B, N, D = x.shape
        
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1).to(x.dtype)
        
        # Reshape for BatchNorm
        x_flat = x.reshape(B * N, D)
        z_flat = self.head(x_flat)
        z = z_flat.reshape(B, N, -1)
        
        return z