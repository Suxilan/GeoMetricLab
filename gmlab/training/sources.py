"""Deterministic training-source interleaving over independent native loaders."""

from collections.abc import Callable, Mapping

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import torch


class MultiSourceLoader:
    """Yield one source's unmodified batch per step, spread across an epoch.

    Each source DataLoader owns its batch, sampler and collate. Its effective
    length already includes Dataset sampling_multiplier and drop_last. All ranks
    must have equal per-source batch counts; differing plans fail before any
    training collective. The epoch getter allows resumed Lightning runs to
    regenerate the same source order at an epoch boundary.
    """

    def __init__(
        self,
        loaders: Mapping[str, DataLoader],
        *,
        epoch_getter: Callable[[], int] | None = None,
        steps_per_source: Mapping[str, int] | None = None,
    ) -> None:
        if not loaders or any(not name or "/" in name for name in loaders):
            raise ValueError("Training sources need unique nonempty path-safe names")
        if any(not isinstance(loader, DataLoader) for loader in loaders.values()):
            raise TypeError("Each training source requires a native DataLoader")
        self.loaders = dict(loaders)
        if steps_per_source is not None and (
            set(steps_per_source) != set(loaders)
            or any(
                type(value) is not int or value < 1
                for value in steps_per_source.values()
            )
        ):
            raise ValueError(
                "steps_per_source must give positive batch counts for every source"
            )
        self.steps_per_source = (
            None if steps_per_source is None else dict(steps_per_source)
        )
        self.epoch_getter = epoch_getter
        self._next_epoch = 0
        self._epoch = None
        self._consumed = dict.fromkeys(self.loaders, 0)
        self._generator_starts = {}

    @property
    def budgets(self) -> dict[str, int]:
        """Return each native loader's effective number of batches on this rank."""
        available = {name: len(loader) for name, loader in self.loaders.items()}
        if self.steps_per_source is None:
            return available
        if any(
            self.steps_per_source[name] > count for name, count in available.items()
        ):
            raise ValueError(
                "steps_per_source exceeds available batches; configure dataset sampling explicitly"
            )
        return dict(self.steps_per_source)

    def __len__(self) -> int:
        return sum(self.budgets.values())

    def state_dict(self) -> dict:
        """Save source position and loader generator starts for deterministic replay."""
        return {
            "epoch": self._epoch,
            "next_epoch": self._next_epoch,
            "budgets": self.budgets,
            "consumed": dict(self._consumed),
            "generator_starts": self._generator_starts,
        }

    def load_state_dict(self, state: dict) -> None:
        """Validate the saved completed prefix before modifying loader state."""
        budgets = self.budgets
        consumed = state.get("consumed", {})
        if state.get("budgets") != budgets or set(consumed) != set(self.loaders):
            raise ValueError("Training source names or budgets changed on resume")
        if any(
            type(consumed[name]) is not int or not 0 <= consumed[name] <= count
            for name, count in budgets.items()
        ):
            raise ValueError("Invalid training source cursor")
        expected = dict.fromkeys(budgets, 0)
        for position, name in enumerate(self._source_order(budgets)):
            if position == sum(consumed.values()):
                break
            expected[name] += 1
        if expected != consumed:
            raise ValueError(
                "Training source cursor is not a completed schedule prefix"
            )
        epoch = state.get("epoch")
        next_epoch = state.get("next_epoch", epoch or 0)
        if (epoch is not None and (type(epoch) is not int or epoch < 0)) or (
            type(next_epoch) is not int or next_epoch < 0
        ):
            raise ValueError("Invalid training source epoch")
        if epoch is None and any(consumed.values()):
            raise ValueError("A nonempty source cursor needs an epoch")
        starts = state.get("generator_starts", {})
        if not isinstance(starts, Mapping) or set(starts) - set(self.loaders):
            raise ValueError("Invalid source generator names")
        for name, value in starts.items():
            if self.loaders[name].generator is None:
                raise ValueError("Saved source generator is missing on resume")
            try:
                torch.Generator().set_state(value.cpu())
            except (AttributeError, TypeError, RuntimeError) as error:
                raise ValueError("Invalid source generator state") from error
        self._epoch, self._next_epoch = epoch, next_epoch
        self._consumed = dict(consumed)
        self._generator_starts = {name: value.clone() for name, value in starts.items()}

    @staticmethod
    def _source_order(budgets):
        """Interleave finite budgets by normalized batch midpoints."""
        consumed = dict.fromkeys(budgets, 0)
        for _ in range(sum(budgets.values())):
            source = min(
                (name for name in budgets if consumed[name] < budgets[name]),
                key=lambda name: (consumed[name] + 0.5) / budgets[name],
            )
            consumed[source] += 1
            yield source

    def _validate_budgets(self):
        budgets = self.budgets
        if any(count < 1 for count in budgets.values()):
            raise ValueError(f"Training source has no batches: {budgets}")
        if dist.is_available() and dist.is_initialized():
            for name, loader in self.loaders.items():
                components = (loader.sampler, loader.batch_sampler)
                if not any(
                    isinstance(component, DistributedSampler)
                    or getattr(component, "distributed_sharding", False)
                    for component in components
                ):
                    raise ValueError(
                        f"Training source {name!r} needs a rank-aware sampler"
                    )
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, budgets)
            if any(other != budgets for other in gathered):
                raise ValueError(
                    f"Training source budgets differ across ranks: {gathered}"
                )
        return budgets

    def _prepare_epoch(self, epoch, budgets):
        resume = self._epoch == epoch and any(self._consumed.values())
        if not resume:
            self._consumed = dict.fromkeys(budgets, 0)
            self._generator_starts = {}
        self._epoch = epoch
        for name, loader in self.loaders.items():
            for sampler in (loader.sampler, loader.batch_sampler):
                if callable(getattr(sampler, "set_epoch", None)):
                    sampler.set_epoch(epoch)
            if loader.generator is not None:
                if resume and name in self._generator_starts:
                    loader.generator.set_state(self._generator_starts[name].cpu())
                elif not resume:
                    self._generator_starts[name] = loader.generator.get_state().clone()

    def __iter__(self):
        """Yield native batches; state snapshots track the yielded schedule prefix."""
        budgets = self._validate_budgets()
        epoch = (
            self.epoch_getter() if self.epoch_getter is not None else self._next_epoch
        )
        self._prepare_epoch(epoch, budgets)
        iterators = {name: iter(loader) for name, loader in self.loaders.items()}
        consumed = dict.fromkeys(budgets, 0)
        replayed = dict(self._consumed)
        for source in self._source_order(budgets):
            try:
                batch = next(iterators[source])
            except StopIteration as error:
                raise RuntimeError(
                    f"Training source {source!r} ended before its budget"
                ) from error
            consumed[source] += 1
            if consumed[source] <= replayed[source]:
                continue
            self._consumed[source] = consumed[source]
            yield {"source": source, "batch": batch}
        self._epoch = None
        self._next_epoch = epoch + 1
        self._consumed = dict.fromkeys(budgets, 0)
