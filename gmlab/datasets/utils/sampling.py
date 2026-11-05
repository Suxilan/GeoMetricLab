"""Explicit DataLoader sampler construction using native constructor contracts."""

import inspect
from collections.abc import Mapping
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Sampler

from ...components import component_definition


class ExactDistributedSampler(Sampler[int]):
    """Partition evaluation rows without padding or repeating sample indices.

    Rank discovery happens at iteration time, after Lightning initializes DDP.
    The same sampler works in single-process evaluation.
    """

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    @staticmethod
    def _partition() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def __iter__(self):
        rank, world_size = self._partition()
        return iter(range(rank, len(self.dataset), world_size))

    def __len__(self) -> int:
        rank, world_size = self._partition()
        return len(range(rank, len(self.dataset), world_size))


class GeoEpochSampler(Sampler):
    """Partition Dataset-owned requests once, with native PyTorch padding.

    Discovery of the process group happens after Lightning initializes DDP.
    Epoch plans use local random generators and never mutate worker RNG state.
    """

    distributed_sharding = True
    supports_sampling_multiplier = True

    def __init__(
        self,
        dataset,
        *,
        seed=0,
        shuffle=True,
        drop_last=False,
        num_replicas=None,
        rank=None,
    ):
        self.dataset = dataset
        self.seed = torch.initial_seed() if seed is None else seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.dataset.epoch_size  # Validate the finite budget before training starts.

    def set_epoch(self, epoch):
        """Select the deterministic Dataset plan for an epoch."""
        self.epoch = epoch

    def _partition(self):
        active = dist.is_available() and dist.is_initialized()
        return DistributedSampler(
            range(self.dataset.epoch_size),
            seed=self.seed,
            shuffle=False,
            drop_last=self.drop_last,
            num_replicas=self.num_replicas
            if self.num_replicas is not None
            else (dist.get_world_size() if active else 1),
            rank=self.rank
            if self.rank is not None
            else (dist.get_rank() if active else 0),
        )

    def __len__(self):
        return len(self._partition())

    def __iter__(self):
        requests = self.dataset.epoch_requests(
            self.epoch, self.seed, shuffle=self.shuffle
        )
        return iter(requests[position] for position in self._partition())


def sampler_definition(
    specification: Mapping[str, Any],
    dataset: Any,
    nested_sampler: Callable[[Mapping[str, Any], Any], Any],
) -> tuple[Callable, dict[str, Any]]:
    """Resolve one loader sampler and its explicit nested ``sampler`` argument.

    PyTorch uses data_source, while project samplers may declare dataset. Inject
    only the declared argument; samplers configured entirely from weights or
    indices receive neither. No other nested mapping is treated as a component.
    The nested callback lets preflight bind signatures without constructing data.
    """
    constructor, arguments = component_definition(specification)
    parameters = inspect.signature(constructor).parameters
    if isinstance(arguments.get("sampler"), Mapping):
        arguments["sampler"] = nested_sampler(arguments["sampler"], dataset)
    injected = "dataset" if "dataset" in parameters else "data_source"
    if injected in parameters:
        if injected in arguments:
            raise ValueError(f"Sampler cannot override injected {injected}")
        arguments[injected] = dataset
    return constructor, arguments


def validate_sampler(specification: Any, dataset: Any = None) -> None:
    """Check explicit sampler signatures recursively without reading a dataset."""
    if specification is None or not isinstance(specification, Mapping):
        return
    constructor, arguments = sampler_definition(
        specification, dataset, validate_sampler
    )
    inspect.signature(constructor).bind(**arguments)


def build_sampler(specification: Any, dataset: Any) -> Any:
    """Construct a configured sampler, or preserve an already constructed object."""
    if not isinstance(specification, Mapping):
        return specification
    constructor, arguments = sampler_definition(specification, dataset, build_sampler)
    return constructor(**arguments)
