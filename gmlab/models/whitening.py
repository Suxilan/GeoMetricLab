"""Fit PCA whitening from feature chunks, independently of extraction and IO."""

import math
from numbers import Integral, Real

import torch


def fit_whitening(chunks, dimension, *, eps=1e-4):
    """Return affine weight [d,D] and bias [d] with sample covariance + ridge.

    Merge centered float64 moments to avoid cancellation and storing [N,D].
    Eigenvectors are selected by descending eigenvalue, matching GeoEncoder's
    existing PCA whitening objective. Memory is bounded by one chunk and [D,D].
    """
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, Integral)
        or dimension < 1
    ):
        raise ValueError("dimension must be a positive integer")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, Real)
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("eps must be finite and positive")
    count, mean, scatter = 0, None, None
    for values in chunks:
        x = torch.as_tensor(values, dtype=torch.float64, device="cpu")
        if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
            raise ValueError("Expected nonempty finite feature chunks [N,D]")
        n = len(x)
        part_mean = x.mean(0)
        centered = x - part_mean
        part_scatter = centered.T @ centered
        if mean is None:
            mean, scatter = part_mean, part_scatter
        else:
            if mean.shape != part_mean.shape:
                raise ValueError("Feature dimensions differ")
            delta = part_mean - mean
            scatter += part_scatter + torch.outer(delta, delta) * (
                count * n / (count + n)
            )
            mean += delta * (n / (count + n))
        count += n
    if count < 2 or dimension > min(count, len(mean)):
        raise ValueError("Insufficient samples or invalid whitening dimension")
    covariance = scatter / (count - 1) + eps * torch.eye(len(mean), dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    indices = torch.argsort(eigenvalues, descending=True)[:dimension]
    matrix = eigenvectors[:, indices] * torch.rsqrt(eigenvalues[indices])
    return matrix.T.float(), (-(mean @ matrix)).float()
