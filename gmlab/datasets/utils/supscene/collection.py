"""Compose scene datasets without changing GeoIDs or GT interpretation."""

import json
from collections.abc import Sequence

import numpy as np
from torch.utils.data import ConcatDataset

from ....components import build_component
from .relations import SceneRelations


def _same_annotations(first, second):
    """Require the same node ordering and sparse GT for a shared GeoID."""
    return first.image_names == second.image_names and all(
        np.array_equal(getattr(first, name), getattr(second, name))
        for name in ("row", "col", "val")
    )


class SubgraphDatasetCollection(ConcatDataset):
    """Concatenate subgraph sources using a shared full-GT query namespace.

    Child sources own image transforms and sampling. They retain their original
    GeoIDs and node IDs. Only GT provenance is rebound to the collection so that
    ranks drawing different sources still resolve one common candidate domain.
    All sources use the collator's declared overlap thresholds. Shared GeoIDs
    must agree on image ordering and annotations; distinct GeoIDs are exclusive.
    """

    def __init__(self, datasets: Sequence):
        if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
            raise TypeError("datasets must be a sequence of subgraph sources")
        sources = [build_component(source) for source in datasets]
        if not sources:
            raise ValueError("At least one subgraph source is required")
        for source in sources:
            if not callable(getattr(source, "relation_resolver", None)):
                raise TypeError("Each subgraph source must provide relation_resolver")
            if not isinstance(getattr(source, "identity_namespace", None), str):
                raise TypeError("Each subgraph source must declare its namespace")
        super().__init__(sources)
        self.identity_namespace = json.dumps(
            ["subgraph-collection", sorted({s.identity_namespace for s in sources})]
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if item.get("exclusive_scenes") is not True:
            raise ValueError("Collection sources must declare exclusive GeoIDs")
        return {**item, "gt_source": self.identity_namespace}

    def relation_resolver(self, *, pos_th=0.3, neg_th=None):
        """Merge static queries once, before workers and distributed execution."""
        graphs = {}
        for source in self.datasets:
            query = source.relation_resolver(pos_th=pos_th, neg_th=neg_th)
            for identity, graph in query.graphs.items():
                previous = graphs.get(identity)
                if previous is not None and not _same_annotations(previous, graph):
                    raise ValueError(
                        "Shared GeoID has conflicting node identities or GT; "
                        "use separate namespaces for unrelated sources"
                    )
                graphs[identity] = graph
        return SceneRelations(
            graphs, pos_th=pos_th, neg_th=neg_th, source=self.identity_namespace
        )
