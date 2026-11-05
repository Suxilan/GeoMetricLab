"""Reusable sample preparation, without owning a reader or training runtime."""

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from torch.utils.data import Dataset

from ...components import build_component
from .identities import integer_tensor
from .samples import validate_modality, validate_sample_row


def _selection_key(row):
    """Identify a candidate's core fields without comparing Tensor values."""
    return (
        id(row["inputs"]),
        row["sample_id"],
        row["modality"],
        "label" in row,
        row.get("label"),
    )


def _same_annotations(original, selected):
    if not original.keys() <= selected.keys():
        return False
    core = {"inputs", "sample_id", "modality", "label"}
    return all(
        selected[name] is value for name, value in original.items() if name not in core
    )


class GeoDataset(Dataset):
    """Prepare selected inputs with shared identity and modality semantics.

    prepare_rows is independent of request shape or sampling algorithm. Readers
    implement load_input(row) when decoding is needed. The optional candidates
    workflow supplies integer-indexed groups and quota selection via __getitem__.
    A Dataset index can denote a single input or an entire group. Neither it nor
    the class label is interpreted as physical sample identity.
    """

    # None declares no class vocabulary; scene identity alone is not a label.
    class_geo_ids: tuple[str, ...] | None = None

    def __init__(
        self,
        *,
        modalities: Sequence[str],
        transform=None,
        sampler=None,
        identity_namespace: str | None = None,
        sampling_multiplier: float = 1.0,
    ):
        super().__init__()
        if (
            isinstance(sampling_multiplier, bool)
            or not isinstance(sampling_multiplier, (int, float))
            or not math.isfinite(sampling_multiplier)
            or sampling_multiplier <= 0
        ):
            raise ValueError("sampling_multiplier must be a finite positive number")
        self.sampling_multiplier = float(sampling_multiplier)
        self.identity_namespace = (
            identity_namespace
            if identity_namespace is not None
            else f"{type(self).__module__}.{type(self).__qualname__}"
        )
        if not isinstance(self.identity_namespace, str) or not self.identity_namespace:
            raise TypeError("identity_namespace must be a nonempty string")
        if isinstance(modalities, (str, bytes)) or not isinstance(modalities, Sequence):
            raise TypeError("modalities must be a sequence of names")
        self.declared_modalities = tuple(validate_modality(name) for name in modalities)
        if not self.declared_modalities or len(set(self.declared_modalities)) != len(
            self.declared_modalities
        ):
            raise ValueError("modalities must be a nonempty sequence of unique names")
        self.transform = transform
        # Resolve from the public configuration on each use. Keeping a second
        # transform map would diverge when callers replace dataset.transform.
        if isinstance(transform, Mapping):
            for name, operation in transform.items():
                if operation is not None and not callable(operation):
                    raise TypeError(f"Transform for {name} must be callable or None")
        for modality in self.declared_modalities:
            self._modality_transform(modality)
        self.candidate_sampler = None if sampler is None else build_component(sampler)
        if self.candidate_sampler is not None and not callable(
            getattr(self.candidate_sampler, "sample", None)
        ):
            raise TypeError("sampler must provide sample(candidates)")

    @property
    def epoch_size(self) -> int:
        """Return the finite exposure budget before distributed padding."""
        return math.ceil(len(self) * self.sampling_multiplier)

    def _epoch_round(self, rng: random.Random, shuffle: bool) -> list:
        """Build one base round using a local random generator."""
        indices = list(range(len(self)))
        if shuffle:
            rng.shuffle(indices)
        return indices

    def epoch_requests(self, epoch: int, seed: int, *, shuffle: bool = True) -> tuple:
        """Expand full rounds and a fresh fractional prefix without replacement.

        The base round preserves the reader's sampling unit. Fractional prefixes
        in different epochs are independent, not complementary selections.
        No global random state is read or modified.
        """
        rng = random.Random(f"{seed}:epoch:{epoch}")
        requests = []
        while len(requests) < self.epoch_size:
            indices = self._epoch_round(rng, shuffle)
            if len(indices) != len(self):
                raise ValueError("Dataset epoch rounds must have a fixed base length")
            requests.extend(indices[: self.epoch_size - len(requests)])
        return tuple(requests)

    def geo_id(self, group: str) -> str:
        """Return a portable group identity, independent of paths and ranks.

        Readers supply stable source group keys, never batch-local positions.
        Sources sharing real identities must share namespace and group keys;
        unrelated sources with reused keys must declare different namespaces.
        Physical sample identity remains separate from this relation scope.
        """
        if not isinstance(group, str) or not group:
            raise ValueError("GeoID group must be a nonempty stable string")
        return json.dumps([self.identity_namespace, group], ensure_ascii=False)

    def _modality_transform(self, modality: str):
        """Resolve the current transform without caching mutable configuration."""
        transform = self.transform
        if isinstance(transform, Mapping):
            if modality not in transform:
                raise ValueError(f"Missing transforms for modalities: {[modality]}")
            transform = transform[modality]
        if transform is not None and not callable(transform):
            raise TypeError(f"Transform for {modality} must be callable or None")
        return transform

    def candidates(self, index: int) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Return ordered, untransformed candidate rows from this source."""
        raise NotImplementedError

    def load_input(self, row: Mapping[str, Any]) -> Any:
        """Load source input; Tensor readers may use this identity implementation."""
        return row["inputs"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        requested = integer_tensor(index, name="Dataset index")
        if requested.ndim:
            raise ValueError("Dataset index must be scalar")
        index = int(requested)
        if not 0 <= index < len(self):
            raise IndexError(f"Dataset index {index} outside [0, {len(self)})")
        candidates = self.candidates(index)
        if not isinstance(candidates, Mapping) or set(candidates) - set(
            self.declared_modalities
        ):
            raise ValueError("Candidates must use only declared modalities")
        ordered = {}
        for modality in self.declared_modalities:
            group = candidates.get(modality, ())
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                raise TypeError("Candidate groups must be explicit row sequences")
            ordered[modality] = []
            for candidate in group:
                row = validate_sample_row(candidate)
                if row["modality"] != modality:
                    raise ValueError("Candidate modality must match its group")
                ordered[modality].append(row)
        legal = {}
        for group in ordered.values():
            for row in group:
                # Snapshot fields before a custom sampler runs; comparing the
                # same mutable row to itself would miss annotation rewrites.
                legal.setdefault(_selection_key(row), []).append(dict(row))
        selected = (
            (row for group in ordered.values() for row in group)
            if self.candidate_sampler is None
            else self.candidate_sampler.sample(ordered)
        )
        return {
            "index": index,
            "rows": self.prepare_rows(self._selected_rows(selected, legal)),
        }

    def _selected_rows(self, selected, legal):
        """Validate candidate provenance lazily, before loading each input."""
        for candidate in selected:
            row = validate_sample_row(candidate)
            originals = legal.get(_selection_key(row), ())
            original = next(
                (value for value in originals if _same_annotations(value, row)), None
            )
            if original is None:
                raise ValueError(
                    "Sampler selected a foreign input or changed candidate identity"
                )
            extra_keys = row.keys() - original.keys()
            if extra_keys - {"is_seed"}:
                raise ValueError("Sampler may only add an is_seed annotation")
            if "is_seed" in row and type(row["is_seed"]) is not bool:
                raise ValueError("is_seed must be boolean")
            yield row

    def prepare_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        load_inputs: bool = True,
    ) -> list[dict[str, Any]]:
        """Prepare already selected rows, independent of request and sampling.

        Each row supplies inputs, a scoped physical sample_id and a declared
        modality. Returned copies preserve annotations and carry row_position,
        the zero-based position in this selection. With load_inputs=False,
        descriptors are validated without decoding or applying transforms.
        The iterable is consumed lazily so selection and augmentation retain
        their established RNG order.
        """
        prepared = []
        for position, candidate in enumerate(rows):
            row = validate_sample_row(candidate)
            if row["modality"] not in self.declared_modalities:
                raise ValueError("Selected row uses an undeclared modality")
            if "row_position" in row:
                declared_position = integer_tensor(
                    row["row_position"], name="row_position", minimum=0
                )
                if declared_position.ndim or int(declared_position) != position:
                    raise ValueError("row_position must match selected-row order")
            row["row_position"] = position
            if load_inputs:
                value = self.load_input(row)
                transform = self._modality_transform(row["modality"])
                row["inputs"] = value if transform is None else transform(value)
            prepared.append(row)
        if not prepared:
            raise ValueError("Selection produced no input rows")
        return prepared
