"""Align feature rows by stable input index without retrieval or GT semantics."""

from collections.abc import Sequence

import torch


def align_embeddings(
    records: Sequence[tuple[torch.Tensor, torch.Tensor]], num_samples: int
) -> torch.Tensor:
    """Restore [N,D] by stable integer index from (indices, embeddings) records.

    DDP padding can repeat an index with equal descriptors. Distinct indices
    remain distinct even for the same physical image. Conflicting duplicates,
    missing/out-of-range indices and non-finite descriptors are errors.
    """
    aligned = {}
    dimension = None
    for indices, embeddings in records:
        indices = torch.as_tensor(indices).cpu()
        embeddings = torch.as_tensor(embeddings).detach().cpu()
        if indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 1:
            raise ValueError("Stable indices must be integer [N]")
        if embeddings.ndim != 2 or len(indices) != len(embeddings):
            raise ValueError("Expected aligned embeddings [N,D]")
        if not torch.isfinite(embeddings).all():
            raise ValueError("Non-finite embeddings")
        if dimension is not None and embeddings.shape[1] != dimension:
            raise ValueError("Inconsistent descriptor dimensions")
        dimension = embeddings.shape[1]
        for index, embedding in zip(indices.tolist(), embeddings):
            if not 0 <= index < num_samples:
                raise ValueError("Sample index outside evaluation manifest")
            if index in aligned:
                if not torch.allclose(aligned[index], embedding, rtol=1e-5, atol=1e-6):
                    raise ValueError(
                        "Conflicting descriptors for duplicate sample index"
                    )
            else:
                aligned[index] = embedding
    if num_samples <= 0 or len(aligned) != num_samples:
        raise ValueError(f"Missing evaluation samples: {len(aligned)}/{num_samples}")
    return torch.stack([aligned[i] for i in range(num_samples)])
