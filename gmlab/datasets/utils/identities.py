"""Validate integer identities and indices before any narrowing conversion."""

from numbers import Integral

import numpy as np
import torch


def _validate_integer_values(values, name, minimum, maximum):
    """Check original scalars/dtypes, including booleans inside mixed containers."""
    if isinstance(values, (torch.Tensor, np.ndarray)):
        if isinstance(values, torch.Tensor):
            try:
                bounds = torch.iinfo(values.dtype)
            except (TypeError, RuntimeError) as error:
                raise ValueError(f"{name} must have an integer dtype") from error
            if values.dtype in (torch.uint16, torch.uint32, torch.uint64):
                # These integer comparisons are unavailable on some backends.
                _validate_integer_values(values.tolist(), name, minimum, maximum)
                return
        else:
            if values.dtype.kind not in "iu":
                raise ValueError(f"{name} must have an integer dtype")
            bounds = np.iinfo(values.dtype)
        count = values.size if isinstance(values, np.ndarray) else values.numel()
        if count == 0:
            return
        # Compare Python bounds before torch can narrow a scalar to the input
        # dtype (e.g. -1 -> uint8(255)). Empty arrays above have no values to reject.
        if minimum > maximum or minimum > bounds.max or maximum < bounds.min:
            raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
        if bounds.max > torch.iinfo(torch.int64).max:
            # Unsigned NumPy values retain their full range as Python integers
            # until they have been checked, before narrowing to signed int64.
            _validate_integer_values(values.tolist(), name, minimum, maximum)
        elif (minimum > bounds.min and bool((values < minimum).any())) or (
            maximum < bounds.max and bool((values > maximum).any())
        ):
            raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
        return
    if isinstance(values, (list, tuple)):
        for value in values:
            _validate_integer_values(value, name, minimum, maximum)
        return
    if isinstance(values, bool) or not isinstance(values, Integral):
        raise ValueError(f"{name} must contain integers, not booleans or floats")
    if not minimum <= values <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")


def integer_tensor(
    values,
    *,
    name: str,
    device: torch.device | str | None = None,
    minimum: int = -(2**63),
    maximum: int = 2**63 - 1,
) -> torch.Tensor:
    """Return exact int64 identities/indices with the source shape preserved.

    Accept integer scalars, rectangular lists/tuples and integer NumPy/Torch
    arrays. Reject floating dtypes (including integral floats), boolean values,
    ragged containers and out-of-range values before casting. Empty lists are
    valid. The caller defines identity scope, unknown sentinels and index bounds.
    """
    _validate_integer_values(values, name, minimum, maximum)
    try:
        return torch.as_tensor(values, device=device, dtype=torch.int64)
    except (ValueError, TypeError, RuntimeError, OverflowError) as error:
        raise ValueError(f"{name} must be rectangular integer data") from error
