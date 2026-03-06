import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLoss(nn.Module):
    def __init__(self, pos_th=0.3, exclude_self=True, eps=1e-8,
                 m=0.4, gamma=80.0):
        super().__init__()
        self.pos_th = float(pos_th)
        self.exclude_self = bool(exclude_self)
        self.eps = float(eps)

        self.m = float(m)
        self.gamma = float(gamma)
        self.softplus = nn.Softplus(beta=1.0)

        self.op = 1.0 + self.m
        self.on = -self.m
        self.delta_p = 1.0 - self.m
        self.delta_n = self.m

    def _logsumexp(self, x: torch.Tensor, mask: torch.Tensor, dim: int, add_one: bool) -> torch.Tensor:
        """
        x:    (..., M)
        mask: (..., M) bool
        return: (..., 1) if keepdim else (...,)
        """
        if add_one:
            shape = list(x.shape)
            shape[dim] = 1
            zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, zeros], dim=dim)
            mask = torch.cat([mask, torch.ones(shape, dtype=torch.bool, device=x.device)], dim=dim)

        x = x.masked_fill(~mask, float("-inf"))
        out = torch.logsumexp(x, dim=dim, keepdim=True)
        out = out.masked_fill(~mask.any(dim=dim, keepdim=True), 0.0)
        return out

    def _compute_loss(self, logits: torch.Tensor, overlap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B,N,N)
        overlap: (B,N,N)
        mask:    (B,N,N) bool
        """
        mask = mask.bool()

        overlap_safe = overlap.masked_fill(~mask, 0.0).clamp_min(0.0)
        pos_mask = mask & (overlap_safe >= self.pos_th)
        neg_mask = mask & (~pos_mask)

        has_pos = pos_mask.any(dim=2)
        has_neg = neg_mask.any(dim=2)
        valid = has_pos & has_neg
        if not valid.any():
            return logits.new_tensor(0.0, requires_grad=True)

        # Build new_mat following PML (only pos/neg positions filled, others 0)
        new_mat = logits.new_zeros(logits.shape)

        # positives
        ap = torch.relu(self.op - logits.detach())
        new_mat = torch.where(
            pos_mask,
            -self.gamma * ap * (logits - self.delta_p),
            new_mat
        )

        # negatives
        an = torch.relu(logits.detach() - self.on)
        new_mat = torch.where(
            neg_mask,
            self.gamma * an * (logits - self.delta_n),
            new_mat
        )

        lse_pos = self._logsumexp(new_mat, pos_mask, dim=2, add_one=False).squeeze(-1)  # (B,N)
        lse_neg = self._logsumexp(new_mat, neg_mask, dim=2, add_one=False).squeeze(-1)  # (B,N)
        losses = self.softplus(lse_pos + lse_neg)

        return losses[valid].mean()

    def forward(self, x: torch.Tensor, overlap: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B,N,D), overlap: (B,N,N), pair_mask: (B,N,N) optional
        """
        B, N, D = x.shape

        logits = torch.matmul(x, x.transpose(1, 2))  # (B,N,N)

        mask = torch.ones((B, N, N), dtype=torch.bool, device=x.device)
        if pair_mask is not None:
            mask = pair_mask.bool()

        if self.exclude_self:
            eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
            mask = mask & ~eye

        return self._compute_loss(logits, overlap, mask)