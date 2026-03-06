import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSimilarityLoss(nn.Module):
    def __init__(self, pos_th=0.25, exclude_self=True, eps=1e-8,
                 alpha=2.0, beta=50.0, base=0.5,
                 rank_weight=10.0, ov_margin=0.05, sim_margin=0.05):
        super().__init__()
        self.pos_th = float(pos_th)
        self.exclude_self = bool(exclude_self)
        self.eps = float(eps)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.base = float(base)
        self.rank_weight = float(rank_weight)
        self.ov_margin = float(ov_margin)
        # backward-compat: if old config passes rank_margin, reuse it as sim_margin
        self.sim_margin = float(sim_margin)

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

    def _compute_msloss(self, logits: torch.Tensor, overlap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
            return (logits * 0.0).sum()

        # PML: pos_exp = margin(logits, base) = base - logits
        #      neg_exp = margin(base, logits) = logits - base
        pos_exp = self.base - logits
        neg_exp = logits - self.base

        pos_loss = (1.0 / self.alpha) * self._logsumexp(
            self.alpha * pos_exp, pos_mask, dim=2, add_one=True
        ).squeeze(-1)  # (B,N)

        neg_loss = (1.0 / self.beta) * self._logsumexp(
            self.beta * neg_exp, neg_mask, dim=2, add_one=True
        ).squeeze(-1)  # (B,N)

        loss = (pos_loss + neg_loss)

        return loss[valid].mean()

    def _compute_rankloss(self, logits: torch.Tensor, overlap: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Continuous pairwise ranking with native MarginRankingLoss.
        If overlap(i,j) - overlap(i,k) > ov_margin, enforce
        sim(i,j) > sim(i,k) + sim_margin.

        logits:  (B,N,N)
        overlap: (B,N,N)
        mask:    (B,N,N) bool
        """
        mask = mask.bool()

        B, N, _ = logits.shape
        logits_flat = logits.view(B * N, N)
        overlap_flat = overlap.view(B * N, N)
        mask_flat = mask.view(B * N, N)

        # diff_ov[row, j, k] = overlap(j) - overlap(k)
        diff_ov = overlap_flat.unsqueeze(2) - overlap_flat.unsqueeze(1)  # (B*N, N, N)
        valid_pairs_mask = (
            mask_flat.unsqueeze(2)
            & mask_flat.unsqueeze(1)
            & (diff_ov > self.ov_margin)
        )

        if not valid_pairs_mask.any():
            return (logits * 0.0).sum()

        x1 = logits_flat.unsqueeze(2).expand(-1, -1, N)[valid_pairs_mask]
        x2 = logits_flat.unsqueeze(1).expand(-1, N, -1)[valid_pairs_mask]
        y = torch.ones_like(x1)

        return F.margin_ranking_loss(x1, x2, y, margin=self.sim_margin, reduction='mean')

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

        ms_loss = self._compute_msloss(logits, overlap, mask)
        rank_loss = self._compute_rankloss(logits, overlap, mask) * self.rank_weight
        return {"main_loss": ms_loss, "aux_loss": rank_loss, "loss": ms_loss + rank_loss}