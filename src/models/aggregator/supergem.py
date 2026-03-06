import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, Dict, Optional, Union
import math
from .base import AggregatorBase

class WeightedGeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True, mode: str = "abs", log_domain: bool = False):
        super().__init__()
        self.eps = float(eps)
        self.mode = mode
        self.log_domain = bool(log_domain)
        p0 = max(float(p), 1.0)
        if learn_p:
            self.p = nn.Parameter(torch.tensor(p0, dtype=torch.float32))
        else:
            self.register_buffer("p", torch.tensor(p0, dtype=torch.float32))

    def _pow_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "clamp":
            return x.clamp_min(self.eps)
        if self.mode == "abs":
            return (x.abs() + self.eps)
        raise ValueError(f"mode must be 'clamp' or 'abs', got {self.mode}")

    def forward(self, a_t: torch.Tensor, x: torch.Tensor, w_sum: torch.Tensor) -> torch.Tensor:
        """
        Calculates GeM magnitude without per-cluster residuals to save memory.
        
        a_t: (B, K, L)   Transposed assignment map
        x:   (B, L, D)   Input features
        w_sum: (B, K, 1) Sum of assignments per cluster
        return: (B, K, D)
        """
        p = self.p.clamp_min(1.0)
        # Compute power on input features once (O(L*D)) instead of per cluster (O(K*L*D))
        # Note: This computes GeM of the original features x, not the residuals (x-c).
        base = self._pow_input(x)

        # Aggregate using matrix multiplication
        if self.log_domain:
            x_pow = torch.exp(p * torch.log(base))
        else:
            x_pow = base.pow(p)
        num = torch.bmm(a_t, x_pow)                         # (B, K, D)
        
        # Normalize by weight sum and take root
        mag = (num / w_sum).pow(1.0 / p)           # (B, K, D)
        return mag


class WeightedDGeM(nn.Module):
    def __init__(self, eps: float = 1e-6, mode: str = "abs", log_domain: bool = False):
        super().__init__()
        self.eps = float(eps)
        self.mode = mode
        self.log_domain = bool(log_domain)

    def _pow_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "clamp":
            return x.clamp_min(self.eps)
        if self.mode == "abs":
            return (x.abs() + self.eps)
        raise ValueError(f"mode must be 'clamp' or 'abs', got {self.mode}")

    def forward(self, a_t: torch.Tensor, x: torch.Tensor, w_sum: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # p: (B, 1, 1) for image or (B, K, 1) for cluster
        p = p.clamp_min(1.0)
        base = self._pow_input(x)

        if p.dim() == 3 and p.size(1) == a_t.size(1) and p.size(1) > 1:
            # cluster-wise p: (B, K, 1)
            B, K, L = a_t.shape
            D = x.size(-1)
            num = x.new_zeros((B, K, D))
            chunk = 8
            for k0 in range(0, K, chunk):
                k1 = min(k0 + chunk, K)
                a_chunk = a_t[:, k0:k1, :]                    # (B, kc, L)
                p_chunk = p[:, k0:k1, :]                      # (B, kc, 1)
                if self.log_domain:
                    x_pow = torch.exp(p_chunk.unsqueeze(-1) * torch.log(base).unsqueeze(1))  # (B, kc, L, D)
                else:
                    x_pow = base.unsqueeze(1).pow(p_chunk.unsqueeze(-1))                      # (B, kc, L, D)
                num[:, k0:k1, :] = (a_chunk.unsqueeze(-1) * x_pow).sum(dim=2)                 # (B, kc, D)
        else:
            # image-wise p: (B, 1, 1)
            if self.log_domain:
                x_pow = torch.exp(p * torch.log(base))
            else:
                x_pow = base.pow(p)
            num = torch.bmm(a_t, x_pow)                         # (B, K, D)

        mag = (num / w_sum).pow(1.0 / p)                        # (B, K, D)
        return mag


class DynamicPHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int, p_min: float, p_max: float):
        super().__init__()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )
        self._init_p_head()

    def _init_p_head(self) -> None:
        # Initialize so p_min + softplus(z) == 3.0 at start.
        target_p = 3.0
        y = max(target_p - self.p_min, 1e-6)
        # softplus^{-1}(y) = log(exp(y) - 1)
        bias = math.log(math.exp(y) - 1.0)
        # Last Linear is index -2 in the Sequential
        last_linear = self.net[-2]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.constant_(last_linear.bias, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p01 = self.net(x)
        return self.p_min + p01

class SuperGeM(AggregatorBase):
    def __init__(
        self,
        in_channels: int,
        num_clusters: int = 64,
        p: float = 3.0,
        eps: float = 1e-6,
        learn_p: bool = True,
        use_dgem: bool = True,
        d_mode: str = "image",  # "image" or "cluster"
        p_min: float = 1.0,
        p_max: float = 8.0,
        p_hidden: int = 64,
        log_domain: bool = True,
    ):
        super().__init__(in_channels=in_channels)
        self.in_channels = int(in_channels)
        self.num_clusters = int(num_clusters)
        self.eps = float(eps)
        self.use_dgem = bool(use_dgem)
        self.d_mode = str(d_mode).lower()
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.p_hidden = int(p_hidden)
        self.log_domain = bool(log_domain)

        self.out_channels = self.num_clusters * self.in_channels

        # Follow NetVLAD/SuperVLAD: Conv2d 1x1 for assignment
        self.assign = nn.Conv2d(self.in_channels, self.num_clusters, kernel_size=1, bias=True)
        
        # In this optimized version, WeightedGeM handles the aggregation logic
        if self.use_dgem:
            self.p_head = DynamicPHead(self.in_channels, self.p_hidden, self.p_min, self.p_max)
            self.wdgem = WeightedDGeM(eps=self.eps, mode="abs", log_domain=self.log_domain)
        else:
            self.wgem = WeightedGeM(p=p, eps=self.eps, learn_p=learn_p, mode="abs", log_domain=self.log_domain)

    def _predict_p(self, x: torch.Tensor, a_t: Optional[torch.Tensor], w_sum: Optional[torch.Tensor]) -> torch.Tensor:
        if self.d_mode == "image":
            # x: (B, C, H, W)
            s = F.adaptive_avg_pool2d(x.abs(), (1, 1)).flatten(1)  # (B, C)
            p = self.p_head(s)
            return p.view(-1, 1, 1)

        if self.d_mode == "cluster":
            if a_t is None or w_sum is None:
                raise ValueError("d_mode='cluster' requires a_t and w_sum")
            # a_t = a_t.detach()
            x_tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
            w = w_sum.clamp_min(self.eps)
            s = torch.bmm(a_t, x_tokens.abs()) / w   # (B, K, C)
            p = self.p_head(s)                       # (B, K, 1)
            return p

        raise ValueError(f"d_mode must be 'image' or 'cluster', got {self.d_mode}")

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        x: (B, C, H, W)
        returns: (B, K*C)
        """
        if x.dim() == 3:
            # (B, N, C) -> (B, C, N) -> (B, C, H, W) where H*W=N
            x = x.transpose(1, 2).unsqueeze(-1)

        if x.dim() != 4:
            raise ValueError(f"x must be 4D (OR 3D), got {tuple(x.shape)}")

        B, C, H, W = x.shape
        if C != self.in_channels:
             raise ValueError(f"dim mismatch: got {C}, expected {self.in_channels}")

        # 1. Assignment a_{k|n}
        # (B, K, H, W)
        a = self.assign(x)
        a = F.softmax(a, dim=1)

        # Flatten spatial dims
        # x_flat: (B, C, N)
        x_flat = x.flatten(2)
        # a_flat: (B, K, N)
        a_flat = a.flatten(2)
        
        # w_sum: (B, K, 1)
        w_sum = a_flat.sum(dim=2).unsqueeze(-1)
        
        # 3. Magnitude (GeM-like)
        # Note: Calculates GeM of original features x, weighted by assignment a_flat
        p_aux = None
        if self.use_dgem:
            p = self._predict_p(x, a_flat, w_sum)
            v = self.wdgem(a_flat, x_flat.transpose(1, 2), w_sum, p=p)     # (B, K, C)
            if p.dim() == 3 and p.size(1) > 1:
                # (B, K, 1) -> (B, 1, K)
                p_aux = p.transpose(1, 2)
            else:
                # (B, 1, 1)
                p_aux = p
        else:
            v = self.wgem(a_flat, x_flat.transpose(1, 2), w_sum)          # (B, K, C)

        # 4. Final Combination
        v = F.normalize(v, p=2, dim=2)                             # (B, K, C)
        v = v.flatten(1)                                         # (B, K*C)

        aux = {"assign_map": a}
        if p_aux is not None:
            aux["p"] = p_aux
        return v, aux

