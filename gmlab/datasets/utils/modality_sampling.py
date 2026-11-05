"""Explicit modal exposure and seed-conditioned candidate selection."""

import random
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any
from types import MappingProxyType

from ...components import build_component, resolve_component
from .identities import integer_tensor
from .samples import validate_modality


class ModalityQuotaSampler:
    """Select per-modality rows without dataset-name or model assumptions.

    Candidate mapping order controls exposure order; quotas only control counts.
    Default sampling uses random.choices, with replacement, lazily per modality.
    candidate_filter(candidates, seed) may select a read-only original-row subset after an
    explicit seed draw. The seed counts toward its modality quota and is marked
    is_seed; it is a sampling anchor, not a declaration of loss-anchor weights.

    missing applies to optional empty modalities. Required or seed modalities
    cannot be absent. With replacement=False, shortage controls insufficient
    populations. Repeated physical identities are not silently deduplicated.
    """

    def __init__(
        self,
        quotas: Mapping[str, int],
        *,
        seed_modality: str | None = None,
        required_modalities: Sequence[str] = (),
        replacement: bool = True,
        missing: str = "skip",
        shortage: str = "error",
        candidate_filter: Callable | str | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(quotas, Mapping) or not quotas:
            raise ValueError("quotas must be a nonempty modality/count mapping")
        self.quotas = {}
        for name, count in quotas.items():
            validate_modality(name)
            number = integer_tensor(count, name=f"quota[{name}]", minimum=0)
            if number.ndim:
                raise ValueError("Each quota must be a scalar integer")
            self.quotas[name] = int(number)
        self.seed_modality = (
            None if seed_modality is None else validate_modality(seed_modality)
        )
        if isinstance(required_modalities, (str, bytes)) or not isinstance(
            required_modalities, Sequence
        ):
            raise TypeError("required_modalities must be a sequence of names")
        self.required_modalities = tuple(
            validate_modality(name) for name in required_modalities
        )
        required = set(self.required_modalities)
        if self.seed_modality is not None:
            required.add(self.seed_modality)
        if any(self.quotas.get(name, 0) < 1 for name in required):
            raise ValueError(
                "Seed and required modalities need positive explicit quotas"
            )
        if type(replacement) is not bool:
            raise TypeError("replacement must be bool")
        if missing not in {"skip", "error"}:
            raise ValueError("missing must be skip or error")
        if shortage not in {"error", "take_available"}:
            raise ValueError("shortage must be error or take_available")
        self.replacement, self.missing, self.shortage = replacement, missing, shortage
        if isinstance(candidate_filter, Mapping):
            self.candidate_filter = build_component(candidate_filter)
        elif isinstance(candidate_filter, str):
            self.candidate_filter = resolve_component(candidate_filter)
        else:
            self.candidate_filter = candidate_filter
        if self.candidate_filter is not None and not callable(self.candidate_filter):
            raise TypeError(
                "candidate_filter must be callable, a component mapping, or an import path"
            )
        if self.candidate_filter is not None and self.seed_modality is None:
            raise ValueError("candidate_filter requires an explicit seed_modality")

    def sample(
        self, candidates: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> Iterator[Mapping[str, Any]]:
        """Yield selected rows; transforms can run before the next modal draw."""
        if not isinstance(candidates, Mapping):
            raise TypeError("candidates must be a modality/row-sequence mapping")
        unknown = set(candidates) - self.quotas.keys()
        if unknown:
            raise ValueError(
                f"Missing explicit quotas for modalities: {sorted(unknown)}"
            )
        pools = {}
        for modality, rows in candidates.items():
            validate_modality(modality)
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise TypeError("Candidate groups must be row sequences")
            pools[modality] = list(rows)
        for modality in self.quotas:
            if modality not in pools:
                pools[modality] = []
        required = set(self.required_modalities)
        if self.seed_modality is not None:
            required.add(self.seed_modality)
        self._validate_available(pools, required)
        seed = None
        if self.seed_modality is not None:
            seed = random.choices(pools[self.seed_modality], k=1)[0]
            if self.candidate_filter is not None:
                pools = self._filter_candidates(pools, seed)
                if not any(row is seed for row in pools[self.seed_modality]):
                    raise ValueError("candidate_filter must retain the selected seed")
                self._validate_available(pools, required)
        for modality, rows in pools.items():
            count = self.quotas[modality]
            if not count or not rows:
                continue
            chosen = []
            if modality == self.seed_modality:
                chosen.append(dict(seed, is_seed=True))
                count -= 1
                if not self.replacement:
                    rows = list(rows)
                    rows.pop(next(i for i, row in enumerate(rows) if row is seed))
            if not self.replacement and count > len(rows):
                if self.shortage == "error":
                    raise ValueError(f"Insufficient candidates for modality {modality}")
                count = len(rows)
            draws = (
                random.choices(rows, k=count)
                if self.replacement
                else random.sample(rows, k=count)
            )
            chosen.extend(draws)
            yield from chosen

    def _filter_candidates(self, pools, seed):
        """Apply a read-only subset filter without drawing or reordering samples."""
        # Protect row fields from accidental filter rewrites. Opaque
        # inputs are shared, not deep-copied into a second data store.
        views, originals = {}, {}
        for modality, rows in pools.items():
            views[modality] = []
            for row in rows:
                view = MappingProxyType(dict(row))
                views[modality].append(view)
                originals[id(view)] = row
        seed_view = next(
            view for view in views[self.seed_modality] if originals[id(view)] is seed
        )
        filtered = self.candidate_filter(
            MappingProxyType({name: tuple(rows) for name, rows in views.items()}),
            seed_view,
        )
        if not isinstance(filtered, Mapping) or set(filtered) - pools.keys():
            raise ValueError("candidate_filter must return existing modality groups")
        selected_pools = {}
        for modality, rows in views.items():
            selection = filtered.get(modality, ())
            if not isinstance(selection, Sequence) or isinstance(
                selection, (str, bytes)
            ):
                raise TypeError("Filtered candidates must be row sequences")
            capacity = Counter(map(id, rows))
            counts = Counter(map(id, selection))
            if any(count > capacity[identity] for identity, count in counts.items()):
                raise ValueError(
                    "candidate_filter cannot introduce or duplicate candidates"
                )
            selected_pools[modality] = [originals[id(view)] for view in selection]
        return selected_pools

    def _validate_available(self, pools, required):
        for modality, rows in pools.items():
            if (
                not rows
                and self.quotas[modality]
                and (modality in required or self.missing == "error")
            ):
                raise ValueError(f"Missing required candidates for modality {modality}")
