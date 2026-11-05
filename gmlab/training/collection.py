"""Detached evaluation collection and weight identity; no optimization state."""

from collections.abc import Callable, Mapping, Sequence

import torch

from ..utils.distributed import broadcast_object, gather_objects, rank
from ..utils.features import align_embeddings


def evaluate_records(
    records: Sequence[tuple[torch.Tensor, torch.Tensor]],
    num_samples: int,
    evaluator: Callable[[torch.Tensor], Mapping[str, float]],
) -> dict[str, float]:
    """Gather variable/empty CPU records, evaluate globally once, broadcast result.

    Object collectives suit this bounded CPU validation path; no GPU throughput
    or large-gallery communication claim is made. All ranks participate even
    with zero local records. Errors are broadcast before raising on every rank.
    Each record contains stable row indices [K] and descriptors [K,D]. Alignment
    removes distributed sampler duplicates and requires all num_samples rows.
    """
    gathered = gather_objects(records)
    result = None
    if rank() == 0:
        try:
            all_records = [
                record for rank_records in gathered for record in rank_records
            ]
            embeddings = align_embeddings(all_records, num_samples)
            metrics = evaluator(embeddings)
            if not isinstance(metrics, Mapping):
                raise TypeError("Retrieval evaluator must return a metric mapping")
            validated = {}
            for name, value in metrics.items():
                if not isinstance(name, str) or not name or name != name.strip():
                    raise ValueError("Retrieval metric names must be nonempty strings")
                scalar = torch.as_tensor(value, dtype=torch.float64)
                if scalar.ndim != 0 or not scalar.isfinite().item():
                    raise ValueError(
                        f"Retrieval metric {name!r} must be a finite scalar"
                    )
                validated[name] = scalar.item()
            result = {"metrics": validated}
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
    result = broadcast_object(result)
    if "error" in result:
        raise ValueError(result["error"])
    return dict(result["metrics"])
