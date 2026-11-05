"""Shared exhaustive descriptor search, independent of retrieval protocols."""

import logging
from collections.abc import Iterator

import torch
from torch.nn import functional as F


log = logging.getLogger(__name__)


def _cpu_faiss_index(faiss, gallery, metric):
    """Build one populated exact index; callers decide failure policy."""
    constructor = faiss.IndexFlatIP if metric == "cosine" else faiss.IndexFlatL2
    index = constructor(gallery.shape[1])
    index.add(gallery.numpy())
    return index


def _faiss_index(gallery, metric, prefer_gpu, gpu_device):
    """Return index, GPU resource owner and backend name, or no index."""
    try:
        import faiss
    except ImportError as error:
        log.warning("FAISS unavailable; using PyTorch: %s", error)
        return None, None, "torch_cpu"
    resources = index = None
    if (
        prefer_gpu
        and hasattr(faiss, "StandardGpuResources")
        and torch.cuda.is_available()
    ):
        try:
            device = torch.cuda.current_device() if gpu_device is None else gpu_device
            resources = faiss.StandardGpuResources()
            resources.setTempMemory(64 * 1024 * 1024)
            constructor = faiss.IndexFlatIP if metric == "cosine" else faiss.IndexFlatL2
            index = faiss.index_cpu_to_gpu(
                resources, device, constructor(gallery.shape[1])
            )
            index.add(gallery.numpy())
            return index, resources, "faiss_gpu"
        except RuntimeError as error:
            log.warning("FAISS GPU initialization failed; trying CPU: %s", error)
            index = resources = None
    try:
        return _cpu_faiss_index(faiss, gallery, metric), None, "faiss_cpu"
    except RuntimeError as error:
        log.warning("FAISS CPU initialization failed; using PyTorch: %s", error)
        return None, None, "torch_cpu"


def _torch_search(queries, gallery, k, metric, tie_break):
    scores = (
        queries @ gallery.T if metric == "cosine" else -torch.cdist(queries, gallery)
    )
    if tie_break == "index":
        return torch.argsort(scores, descending=True, stable=True)[:, :k].cpu()
    return scores.topk(k, dim=1).indices.cpu()


def exact_search_chunks(
    queries,
    gallery,
    *,
    k: int,
    metric: str = "cosine",
    backend: str = "auto",
    tie_break: str = "native",
    chunk_size: int = 128,
    faiss_threshold: int = 4000,
    prefer_gpu: bool = True,
    gpu_device: int | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield (query offset, CPU int64 gallery indices [B,k]).

    Inputs are finite [Q,D]/[G,D] descriptors. Search is exhaustive, never ANN.
    Auto uses FAISS above the gallery threshold. GPU failures retry on CPU
    FAISS; unavailable or failed CPU FAISS falls back to CPU PyTorch. Explicit
    torch/index tie policies pin PyTorch on the input device. Native ties depend
    on the backend. Query chunks bound temporary scores/results, not gallery
    storage. Computation uses FP32. GPU index resources live until exhaustion or
    generator close; fallback releases them before constructing the CPU index.
    """
    if backend not in {"auto", "torch", "faiss"}:
        raise ValueError("backend must be auto, torch or faiss")
    if metric not in {"cosine", "l2"} or tie_break not in {"native", "torch", "index"}:
        raise ValueError("Unknown search metric or tie policy")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if type(faiss_threshold) is not int or faiss_threshold < 0:
        raise ValueError("faiss_threshold must be a nonnegative integer")
    if type(prefer_gpu) is not bool:
        raise TypeError("prefer_gpu must be bool")
    if gpu_device is not None and (type(gpu_device) is not int or gpu_device < 0):
        raise ValueError("gpu_device must be a nonnegative integer or null")
    q = torch.as_tensor(queries).detach().float()
    g = torch.as_tensor(gallery).detach().float()
    if (
        q.ndim != 2
        or g.ndim != 2
        or q.shape[1] != g.shape[1]
        or g.shape[1] == 0
        or q.device != g.device
        or not torch.isfinite(q).all()
        or not torch.isfinite(g).all()
    ):
        raise ValueError("Expected finite [Q,D]/[G,D] on the same device, with D>0")
    if type(k) is not int or not 1 <= k <= len(g):
        raise ValueError("k must be between one and the gallery size")
    if metric == "cosine":
        q, g = F.normalize(q, dim=-1), F.normalize(g, dim=-1)
    use_faiss = backend == "faiss" or (backend == "auto" and len(g) > faiss_threshold)
    if tie_break != "native":
        if backend == "faiss":
            raise ValueError("FAISS requires tie_break=native")
        use_faiss = False
    index = resources = None
    selected = f"torch_{q.device.type}"
    if use_faiss:
        q, g = q.cpu().contiguous(), g.cpu().contiguous()
        index, resources, selected = _faiss_index(g, metric, prefer_gpu, gpu_device)
    log.info(
        "Exact search: backend=%s queries=%d gallery=%d k=%d tie_break=%s",
        selected,
        len(q),
        len(g),
        k,
        tie_break,
    )
    try:
        for start in range(0, len(q), chunk_size):
            batch = q[start : start + chunk_size]
            while index is not None:
                try:
                    _, rows = index.search(batch.numpy(), k)
                    result = torch.from_numpy(rows)
                    break
                except RuntimeError as error:
                    log.warning("%s search failed; falling back: %s", selected, error)
                    was_gpu = selected == "faiss_gpu"
                    index = resources = None
                    selected = "torch_cpu"
                    if was_gpu:
                        index, resources, selected = _faiss_index(
                            g, metric, False, None
                        )
            else:
                result = _torch_search(batch, g, k, metric, tie_break)
            yield start, result
    finally:
        del index, resources
