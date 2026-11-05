"""Small shared inference precision helpers using native PyTorch autocast."""

from contextlib import contextmanager
from collections.abc import Iterator, Mapping
from typing import Any

import torch


def resolve_inference_precision(
    device: torch.device | str,
    precision: str | int | None = None,
    *,
    matmul_tf32: bool | None = None,
    cudnn_tf32: bool | None = None,
) -> dict[str, Any]:
    """Resolve one explicit numerical policy, independently of storage dtype.

    Supports FP32 weights/inputs with 32-true or mixed autocast. CPU supports
    bf16-mixed, CUDA supports both mixed dtypes when available. CPU has no TF32;
    an explicit request to enable it is rejected. CUDA defaults inherit current
    flags. Policies contain no device names, rank count or loader settings.
    """
    device = torch.device(device)
    precision = "32-true" if precision is None else precision
    precision = "32-true" if precision in (32, "32") else precision
    if precision not in ("32-true", "bf16-mixed", "16-mixed"):
        raise ValueError(f"Unsupported inference precision: {precision}")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Inference precision supports only CPU/CUDA")
    for name, value in (("matmul_tf32", matmul_tf32), ("cudnn_tf32", cudnn_tf32)):
        if value is not None and type(value) is not bool:
            raise ValueError(f"{name} must be boolean or None")
    if device.type == "cpu":
        if precision == "16-mixed":
            raise ValueError("16-mixed inference requires CUDA; use bf16-mixed on CPU")
        if matmul_tf32 or cudnn_tf32:
            raise ValueError("TF32 inference requires CUDA")
        matmul, convolution = False, False
    else:
        if not torch.cuda.is_available():
            raise ValueError("CUDA inference requested but unavailable")
        with torch.cuda.device(device):
            if precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
                raise ValueError(
                    "This CUDA device does not support bf16-mixed inference"
                )
        matmul = (
            torch.backends.cuda.matmul.allow_tf32
            if matmul_tf32 is None
            else matmul_tf32
        )
        convolution = (
            torch.backends.cudnn.allow_tf32 if cudnn_tf32 is None else cudnn_tf32
        )
        if matmul_tf32 is None and torch.get_float32_matmul_precision() == "medium":
            raise ValueError(
                "Set matmul_tf32 explicitly instead of inheriting medium matmul precision"
            )
    return {
        "precision": precision,
        "matmul_tf32": matmul,
        "cudnn_tf32": convolution,
        "float32_matmul_precision": "high" if matmul else "highest",
    }


def trainer_inference_options(
    trainer: Any,
    *,
    precision: str | int | None = None,
    matmul_tf32: bool | None = None,
    cudnn_tf32: bool | None = None,
) -> dict[str, Any]:
    """Inherit the native Trainer precision, rejecting a conflicting inner request."""
    actual = trainer.precision
    if (
        precision is not None
        and ("32-true" if precision in (32, "32") else precision) != actual
    ):
        raise ValueError(
            f"Inference precision {precision} conflicts with Trainer precision {actual}"
        )
    policy = resolve_inference_precision(
        trainer.strategy.root_device,
        actual,
        matmul_tf32=matmul_tf32,
        cudnn_tf32=cudnn_tf32,
    )
    # Pass resolved flags explicitly to spawned workers rather than depending on
    # their freshly initialized PyTorch globals.
    return {key: policy[key] for key in ("precision", "matmul_tf32", "cudnn_tf32")}


@contextmanager
def inference_precision_context(
    device: torch.device | str, policy: Mapping
) -> Iterator[None]:
    """Apply autocast/TF32 for one inference scope and restore caller settings."""
    device = torch.device(device)
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        torch.set_float32_matmul_precision(policy["float32_matmul_precision"])
        torch.backends.cudnn.allow_tf32 = policy["cudnn_tf32"]
        mixed = policy["precision"] != "32-true"
        dtype = torch.bfloat16 if policy["precision"] == "bf16-mixed" else torch.float16
        with torch.autocast(device.type, dtype=dtype, enabled=mixed):
            yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        if torch.backends.cuda.matmul.allow_tf32 != previous[1]:
            torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


def inference_precision_record(
    device: torch.device | str, policy: Mapping
) -> dict[str, Any]:
    """Read actual active autocast/TF32 state while inside the inference scope."""
    enabled = torch.is_autocast_enabled(torch.device(device).type)
    return {
        **policy,
        "autocast_enabled": enabled,
        "autocast_dtype": (
            str(torch.get_autocast_dtype(torch.device(device).type))
            if enabled
            else None
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
