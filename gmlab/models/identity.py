"""Actual registered model state identity, independent of construction/loading."""

import hashlib

import torch


class ModelIdentityUnavailable(TypeError):
    """Registered state has no supported stable cache identity representation."""


def model_digest(model: torch.nn.Module) -> str:
    """Hash names, shapes, dtype and bytes of parameters and all buffers.

    Include nonpersistent registered buffers too: checkpoint serialization policy
    does not decide whether a buffer affects inference. Non-Tensor extra_state
    needs an explicit future contract rather than a repr-based fingerprint.
    """
    state = dict(model.state_dict())
    state.update(dict(model.named_buffers()))
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            raise ModelIdentityUnavailable(
                f"Model identity requires Tensor state/registered buffers: {name}"
            )
        value = value.detach().cpu().contiguous()
        digest.update(str((name, tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_identity_record(model: torch.nn.Module) -> dict[str, str | None]:
    """Describe optional training provenance without imposing cache eligibility.

    Native PyTorch extra_state may be checkpointable yet have no supported
    stable fingerprint. Such a model can train, validate and resume; its record
    explicitly lacks a digest. Cache producers still call strict model_digest.
    Other state serialization or tensor failures propagate instead of being
    misreported as an unsupported identity.
    """
    try:
        return {"sha256": model_digest(model), "unavailable_reason": None}
    except ModelIdentityUnavailable as error:
        return {"sha256": None, "unavailable_reason": str(error)}
