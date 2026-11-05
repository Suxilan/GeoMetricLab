"""Tensor communication without GT, objective, trainer or storage semantics."""

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

GatherFunction = Callable[..., torch.Tensor]
_runtime_gather: ContextVar[GatherFunction | None] = ContextVar(
    "gmlab_runtime_gather", default=None
)


@contextmanager
def gather_context(gather_fn: GatherFunction) -> Iterator[None]:
    """Bind a runtime's tensor gather for this call and its nested objectives.

    The function must return stacked equal-shaped rank tensors and honor
    sync_grads. Tokens restore outer calls and exceptions; runtime callables
    never become module/checkpoint state. Explicit component injection wins.
    """
    if not callable(gather_fn):
        raise TypeError("gather_fn must be callable")
    token = _runtime_gather.set(gather_fn)
    try:
        yield
    finally:
        _runtime_gather.reset(token)


def current_gather_function() -> GatherFunction | None:
    """Return the enclosing runtime gather, or None for independent execution."""
    return _runtime_gather.get()


def distributed_active() -> bool:
    """Return whether a multi-rank process group is initialized."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    """Return the current global rank, or zero without a process group."""
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def world_size() -> int:
    """Return the process count, or one without a process group."""
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def gather_objects(value: Any, *, destination: int = 0) -> list[Any] | None:
    """Gather trusted, picklable rank objects only to the destination rank.

    This detached, whole-object operation is for bounded CPU records/control
    messages, not large feature-map export. All ranks must participate.
    """
    if not distributed_active():
        return [value]
    gathered = [None] * world_size() if rank() == destination else None
    dist.gather_object(value, gathered, dst=destination)
    return gathered


def broadcast_object(value: Any, *, source: int = 0) -> Any:
    """Broadcast one trusted, picklable object from source to every rank."""
    if not distributed_active():
        return value
    values = [value]
    dist.broadcast_object_list(values, src=source)
    return values[0]


def gather_tensor(
    tensor: torch.Tensor,
    *,
    gather_fn: GatherFunction | None = None,
    enabled: bool = True,
) -> torch.Tensor:
    """Stack equal-shaped tensors as [world_size,...] without an autograd edge.

    All ranks participate in identical order. The caller selects a tensor device
    compatible with its backend; CPU is valid for Gloo and CUDA for NCCL.
    """
    with torch.no_grad():
        tensor = tensor.detach().contiguous()
        if not enabled or not distributed_active():
            return tensor.unsqueeze(0)
        if gather_fn is not None:
            return gather_fn(tensor, sync_grads=False)
        return torch.stack(all_gather(tensor))


def gather_tensor_with_grad(
    tensor: torch.Tensor,
    *,
    gather_fn: GatherFunction | None = None,
    enabled: bool = True,
) -> torch.Tensor:
    """Stack equal-shaped tensors while retaining remote autograd contributions.

    Every participating rank must traverse backward through this gather in the
    same order, even when its local objective is zero. No DDP loss scaling or
    reduction denominator is applied here.
    """
    tensor = tensor.contiguous()
    if not enabled or not distributed_active():
        return tensor.unsqueeze(0)
    if gather_fn is not None:
        return gather_fn(tensor, sync_grads=True)
    return torch.stack(all_gather(tensor))


def _gather_variable(
    tensor: torch.Tensor,
    sizes: Sequence[int],
    *,
    square: bool,
    gather: Callable[[torch.Tensor], torch.Tensor],
) -> list[torch.Tensor]:
    padding = max(sizes) - tensor.shape[0]
    if square:
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("square gather requires a square matrix")
        padded = F.pad(tensor, (0, padding, 0, padding))
    else:
        padded = F.pad(tensor, (0, 0) * (tensor.ndim - 1) + (0, padding))
    gathered = gather(padded)
    if square:
        return [part[:size, :size] for part, size in zip(gathered, sizes)]
    return [part[:size] for part, size in zip(gathered, sizes)]


def gather_variable_rows(
    tensor: torch.Tensor,
    sizes: Sequence[int],
    *,
    square: bool = False,
    gather_fn: GatherFunction | None = None,
    enabled: bool = True,
) -> list[torch.Tensor]:
    """Gather rank-ordered variable [N,...] tensors without gradients.

    sizes contains each rank's leading dimension; trailing layout/dtype must
    agree. square=True pads/unpads both axes of a square matrix. This function
    replicates all rows on each rank and is not a bounded export writer.
    """
    gather = partial(gather_tensor, gather_fn=gather_fn, enabled=enabled)
    return _gather_variable(tensor, sizes, square=square, gather=gather)


def gather_variable_rows_with_grad(
    tensor: torch.Tensor,
    sizes: Sequence[int],
    *,
    square: bool = False,
    gather_fn: GatherFunction | None = None,
    enabled: bool = True,
) -> list[torch.Tensor]:
    """Gather variable [N,...] tensors with remote gradients, without loss scaling.

    Layout and participation follow gather_variable_rows; backward participation
    follows gather_tensor_with_grad. Returned parts remain rank ordered.
    """
    gather = partial(gather_tensor_with_grad, gather_fn=gather_fn, enabled=enabled)
    return _gather_variable(tensor, sizes, square=square, gather=gather)


def check_rank_consistency(error=None, signature=None, *, enabled=True):
    """Raise a controlled rank failure on all participants before payload transfer."""
    active = enabled and distributed_active()
    local = {"error": error, "signature": signature}
    messages = gather_objects(local) if active else [local]
    failure = None
    if not active or rank() == 0:
        failure = next(
            (
                f"rank {i}: {message['error']}"
                for i, message in enumerate(messages)
                if message["error"]
            ),
            None,
        )
        if failure is None and any(
            message["signature"] != messages[0]["signature"] for message in messages
        ):
            failure = "Rank operation contracts differ"
    failure = broadcast_object(failure) if active else failure
    if failure:
        exceptions = {
            "ValueError": ValueError,
            "TypeError": TypeError,
            "FileExistsError": FileExistsError,
            "FileNotFoundError": FileNotFoundError,
        }
        error_name = (
            failure.split(": ", 2)[1] if failure.startswith("rank ") else "RuntimeError"
        )
        raise exceptions.get(error_name, RuntimeError)(failure)
