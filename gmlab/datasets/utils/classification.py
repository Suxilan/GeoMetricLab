"""Compose class-supervised sources with one stable PML label vocabulary."""

from bisect import bisect_right
from collections.abc import Mapping, Sequence

from torch.utils.data import ConcatDataset

from ...components import build_component
from .identities import integer_tensor


class ClassDatasetCollection(ConcatDataset):
    """Concatenate canonical row datasets and remap labels by declared GeoID.

    Each source exposes class_geo_ids in local label order and returns items
    containing rows. Labels are contiguous across the entire collection, never
    assigned from batch contents. Equal GeoIDs share a class across sources;
    unrelated sources must use different identity namespaces. Input datasets
    and their rows are not mutated. Vocabulary order is lexicographic in GeoID,
    independent of source order, workers and ranks. Changing the vocabulary
    requires remapping any existing proxy-based classifier weights.
    """

    def __init__(self, datasets: Sequence):
        if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
            raise TypeError("datasets must be a sequence of class-supervised sources")
        sources = [build_component(source) for source in datasets]
        class_lists = []
        for source in sources:
            identities = getattr(source, "class_geo_ids", None)
            if not isinstance(identities, tuple) or not identities:
                raise ValueError(
                    "Each source must declare a nonempty class_geo_ids tuple"
                )
            if any(not isinstance(key, str) or not key for key in identities):
                raise ValueError("Class GeoIDs must be nonempty strings")
            if len(set(identities)) != len(identities):
                raise ValueError("A source cannot declare duplicate class GeoIDs")
            class_lists.append(identities)
        if not sources:
            raise ValueError("At least one class-supervised source is required")
        super().__init__(sources)
        identities = {key for keys in class_lists for key in keys}
        self.class_geo_ids = tuple(sorted(identities))
        vocabulary = {key: index for index, key in enumerate(self.class_geo_ids)}
        self.label_maps = [
            tuple(vocabulary[key] for key in keys) for keys in class_lists
        ]
        self.num_classes = len(vocabulary)

    def __getitem__(self, index):
        value = integer_tensor(index, name="Dataset index")
        if value.ndim:
            raise ValueError("Dataset index must be scalar")
        index = int(value)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Dataset index is outside the collection")
        source_index = bisect_right(self.cumulative_sizes, index)
        offset = 0 if source_index == 0 else self.cumulative_sizes[source_index - 1]
        item = self.datasets[source_index][index - offset]
        if not isinstance(item, Mapping) or "rows" not in item:
            raise ValueError(
                "Class sources must return canonical items containing rows"
            )
        rows = [self._remap_row(row, source_index) for row in item["rows"]]
        return {**item, "index": index, "rows": rows}

    def _remap_row(self, row, source_index):
        mapping = self.label_maps[source_index]
        label = integer_tensor(
            row["label"], name="Class label", minimum=0, maximum=len(mapping) - 1
        )
        if label.ndim:
            raise ValueError("Each row must have a scalar class label")
        local_label = int(label)
        geo_id = self.datasets[source_index].class_geo_ids[local_label]
        if row.get("geo_id", geo_id) != geo_id:
            raise ValueError("Row GeoID contradicts its declared class label")
        return {**row, "geo_id": geo_id, "label": mapping[local_label]}
