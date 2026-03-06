"""SupScene对比损失函数：SupConLoss"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class SupConLoss(nn.Module):
    def __init__(self, tau=0.1, mode="hard", gamma=2.0, pos_th=0.25,
                 exclude_self=True, eps=1e-8):
        super().__init__()
        assert mode in ("hard", "soft")
        self.tau = float(tau)
        self.mode = mode
        self.gamma = float(gamma)
        self.pos_th = float(pos_th)
        self.exclude_self = bool(exclude_self)
        self.eps = float(eps)

    def _logsumexp(self, x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
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

        # 关键：先把无效位置 overlap 置零，并裁掉负值，避免 pow 产生 NaN/Inf
        overlap_safe = overlap.masked_fill(~mask, 0.0).clamp_min(0.0)

        valid_pos_mask = (overlap_safe >= self.pos_th)
        if self.mode == "hard":
            W = valid_pos_mask.float()
        else:
            # soft mode: only valid positives contribute to numerator.
            # samples below pos_th stay in denominator as negatives via mask,
            # but get zero positive weight in W.
            W = torch.where(valid_pos_mask, overlap_safe.pow(self.gamma), torch.zeros_like(overlap_safe))

        W = W * mask.float()

        has_any = mask.any(dim=2)           # (B,N)
        has_pos = (W > 0).any(dim=2)        # (B,N)
        valid = has_any & has_pos
        if not valid.any():
            # Keep graph connected to avoid DDP unused-parameter/reduction issues.
            return (logits * 0.0).sum()

        # masked log-softmax with stability
        logits_masked = logits.masked_fill(~mask, float("-inf"))
        row_max = logits_masked.max(dim=2, keepdim=True).values
        row_max = torch.where(has_any.unsqueeze(-1), row_max, torch.zeros_like(row_max))
        logits_stable = torch.where(has_any.unsqueeze(-1), logits - row_max.detach(), torch.zeros_like(logits))

        denom = self._logsumexp(logits_stable, mask, dim=2)          # (B,N,1)
        log_prob = logits_stable - denom
        log_prob = log_prob.masked_fill(~mask, 0.0)

        Z = (W.sum(dim=2) + self.eps).clamp_min(1e-8)               # (B,N)
        per_anchor = -(W * log_prob).sum(dim=2) / Z                 # (B,N)

        return per_anchor[valid].mean()

    def forward(self, x: torch.Tensor, overlap: torch.Tensor, pair_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B,N,D), overlap: (B,N,N), pair_mask: (B,N,N) optional
        """
        B, N, D = x.shape
        x = F.normalize(x, p=2, dim=-1, eps=1e-6)

        logits = torch.matmul(x, x.transpose(1, 2)) / max(self.tau, 1e-6)   # (B,N,N)

        mask = torch.ones((B, N, N), dtype=torch.bool, device=x.device)
        if pair_mask is not None:
            mask = pair_mask.bool()

        if self.exclude_self:
            eye = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
            mask = mask & ~eye

        return self._compute_loss(logits, overlap, mask)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from typing import Optional, List, Tuple


# class SupConLoss(nn.Module):
#     def __init__(
#         self,
#         tau=0.1,
#         mode="hard",          # 建议先用 hard 排除 soft 的数值问题
#         pos_th=0.3,
#         gamma=0.7,
#         exclude_self=True,
#         eps=1e-8,
#         use_cross_batch=True,
#         use_cross_device=True,
#         use_batch_hard=False,
#         hard_neg_k=1024,
#     ):
#         super().__init__()
#         assert mode in ("hard", "soft")
#         self.tau = float(tau)
#         self.mode = mode
#         self.pos_th = float(pos_th)
#         self.exclude_self = bool(exclude_self)
#         self.eps = float(eps)

#         self.use_cross_batch = bool(use_cross_batch)
#         self.use_cross_device = bool(use_cross_device)

#         self.use_batch_hard = bool(use_batch_hard)
#         self.hard_neg_k = int(hard_neg_k)

#     def _logsumexp(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         x = x.masked_fill(~mask, float("-inf"))
#         out = torch.logsumexp(x, dim=1, keepdim=True)
#         return out.masked_fill(~mask.any(dim=1, keepdim=True), 0.0)

#     def _pack_masks_local(self, overlap: torch.Tensor, pair_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
#         # overlap: (B,N,N), pair_mask: (B,N,N) or None
#         B, N, _ = overlap.shape
#         BN = B * N
#         device = overlap.device

#         o_full = overlap.new_zeros((BN, BN))
#         if self.use_cross_batch:
#             m_full = torch.ones((BN, BN), dtype=torch.bool, device=device)
#         else:
#             m_full = torch.zeros((BN, BN), dtype=torch.bool, device=device)

#         for b in range(B):
#             i0, i1 = b * N, (b + 1) * N
#             o_full[i0:i1, i0:i1] = overlap[b]
#             if pair_mask is None:
#                 m_full[i0:i1, i0:i1] = True
#             else:
#                 m_full[i0:i1, i0:i1] = pair_mask[b].bool()

#         if self.exclude_self:
#             idx = torch.arange(BN, device=device)
#             m_full[idx, idx] = False

#         return o_full, m_full  # (BN,BN), (BN,BN)

#     def _gather_all_keys(self, x_local: torch.Tensor) -> Tuple[torch.Tensor, int, List[int]]:
#         if not (self.use_cross_device and dist.is_available() and dist.is_initialized()):
#             return x_local, 0, [x_local.size(0)]

#         rank = dist.get_rank()
#         world = dist.get_world_size()
#         device = x_local.device
#         D = x_local.size(1)

#         n_local = x_local.size(0)
#         n_t = torch.tensor([n_local], device=device, dtype=torch.long)
#         n_list = [torch.zeros_like(n_t) for _ in range(world)]
#         dist.all_gather(n_list, n_t)
#         sizes = [int(t.item()) for t in n_list]
#         max_n = max(sizes)

#         if n_local < max_n:
#             pad = x_local.new_zeros((max_n - n_local, D))
#             x_pad = torch.cat([x_local, pad], dim=0)
#         else:
#             x_pad = x_local

#         gathered = [x_pad.new_zeros(x_pad.shape) for _ in range(world)]
#         dist.all_gather(gathered, x_pad.detach())  # 远端 stop-grad
#         gathered[rank] = x_pad                     # 本端保梯度

#         x_all = torch.cat([g[:sizes[i]] for i, g in enumerate(gathered)], dim=0)
#         offset = sum(sizes[:rank])
#         return x_all, offset, sizes

#     def _compute_loss(self, logits: torch.Tensor, overlap_row: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         mask = mask.bool()

#         # 先 hard，减少不稳定源（soft 先别开）
#         pos_mask = mask & (overlap_row >= self.pos_th)
#         neg_mask = mask & (~pos_mask)

#         has_pos = pos_mask.any(dim=1)
#         has_neg = neg_mask.any(dim=1)
#         if not (has_pos & has_neg).any():
#             return logits.new_tensor(0.0, requires_grad=True)

#         if self.use_batch_hard:
#             k = min(self.hard_neg_k, logits.size(1))
#             neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
#             vals, idx = torch.topk(neg_logits, k=k, dim=1)
#             hard_neg = torch.zeros_like(neg_mask)
#             hard_neg.scatter_(1, idx, vals > float("-inf"))
#             keep = pos_mask | (hard_neg & neg_mask)
#         else:
#             keep = mask

#         valid = keep.any(dim=1) & pos_mask.any(dim=1) & (keep & ~pos_mask).any(dim=1)
#         if not valid.any():
#             return logits.new_tensor(0.0, requires_grad=True)

#         # masked log-softmax（稳定）
#         logits_masked = logits.masked_fill(~keep, float("-inf"))
#         row_max = logits_masked.max(dim=1, keepdim=True).values
#         row_max = torch.where(keep.any(dim=1, keepdim=True), row_max, torch.zeros_like(row_max))
#         logits_stable = torch.where(keep.any(dim=1, keepdim=True), logits - row_max.detach(), torch.zeros_like(logits))

#         denom = self._logsumexp(logits_stable, keep)
#         log_prob = (logits_stable - denom).masked_fill(~keep, 0.0)

#         W = pos_mask.float()  # hard supcon：正样本权重=1
#         Z = (W.sum(dim=1) + self.eps).clamp_min(1e-8)
#         per_anchor = -(W * log_prob).sum(dim=1) / Z
#         return per_anchor[valid].mean()

#     def forward(self, x: torch.Tensor, overlap: torch.Tensor, pair_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # x: (B,N,D), overlap: (B,N,N)
#         B, N, D = x.shape
#         x_local = F.normalize(x, p=2, dim=-1, eps=1e-6).reshape(B * N, D)  # (Q,D)
#         Q = x_local.size(0)

#         o_full, m_full = self._pack_masks_local(overlap, pair_mask)        # (Q,Q), (Q,Q)

#         x_all, offset, _ = self._gather_all_keys(x_local)                  # (K,D)
#         K = x_all.size(0)

#         logits = (x_local @ x_all.t()) / max(self.tau, 1e-6)               # (Q,K)

#         overlap_row = x_local.new_zeros((Q, K))
#         mask = torch.ones((Q, K), dtype=torch.bool, device=x_local.device)

#         end = min(offset + Q, K)
#         w = end - offset
#         if w > 0:
#             overlap_row[:, offset:end] = o_full[:, :w]
#             mask[:, offset:end] = m_full[:, :w]

#         # m_full 已经处理了 exclude_self；这里不需要再 mask 一次
#         return self._compute_loss(logits, overlap_row, mask)
