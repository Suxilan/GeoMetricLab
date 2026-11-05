"""Distributed scalar reduction and its small collective adapter.

Generic detached/autograd communication lives in utils.distributed. Training
candidate collection lives in training.collectives. This module owns exact
sum/count reporting and DDP backward normalization, without any GT semantics.
"""

from collections.abc import Sequence

import torch
import torch.distributed as dist

from ..utils.distributed import (
    GatherFunction,
    current_gather_function,
    distributed_active,
    gather_tensor,
    gather_tensor_with_grad,
    gather_variable_rows,
    gather_variable_rows_with_grad,
)


class MetricCollectives:
    """Use explicit injection, then scoped runtime gather, then native transport.

    MetricModule binds Lightning's all_gather for its criterion call. Components
    outside that scope remain usable in an independent PyTorch training loop.
    """

    def __init__(self, enabled: bool, gather_fn: GatherFunction | None = None) -> None:
        self.active = enabled and distributed_active()
        self.rank = dist.get_rank() if self.active else 0
        self.world_size = dist.get_world_size() if self.active else 1
        self._gather_fn = (
            gather_fn if gather_fn is not None else current_gather_function()
        )

    def gather_equal(
        self, tensor: torch.Tensor, *, sync_grads: bool = False
    ) -> torch.Tensor:
        """Stack equal-shaped rank tensors as [world_size,...].

        sync_grads=True retains remote contributions in autograd; otherwise the
        gather is detached. All ranks must use the same shape and call order.
        """
        if not self.active:
            return tensor.unsqueeze(0)
        gather = gather_tensor_with_grad if sync_grads else gather_tensor
        return gather(tensor, gather_fn=self._gather_fn, enabled=self.active)

    def sizes(self, local_size: int, device: torch.device) -> list[int]:
        size = torch.tensor([local_size], device=device, dtype=torch.long)
        return self.gather_equal(size).flatten().tolist()

    def gather_variable(
        self,
        tensor: torch.Tensor,
        sizes: Sequence[int],
        *,
        sync_grads: bool = False,
        square: bool = False,
    ) -> list[torch.Tensor]:
        """Pad, gather and unpad each rank's [N,...] tensor using rank-ordered sizes.

        square=True gathers [N,N] relation blocks and removes padding on both
        axes. It does not create observed relations between those blocks.
        """
        gather = gather_variable_rows_with_grad if sync_grads else gather_variable_rows
        return gather(
            tensor,
            sizes,
            square=square,
            gather_fn=self._gather_fn,
            enabled=self.active,
        )
