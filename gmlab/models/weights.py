"""Strict model-only weight loading, independent of training run policy."""

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def validate_initialization(initialization: Mapping | None) -> dict:
    """Validate model-only initialization options before opening weight files."""
    options = dict(initialization or {})
    allowed = {"checkpoint", "strict", "strip_prefixes", "rename_prefixes"}
    if options.keys() - allowed:
        raise ValueError(
            "initialization accepts checkpoint, strict, strip_prefixes and rename_prefixes"
        )
    if type(options.get("strict", True)) is not bool:
        raise ValueError("initialization.strict must be boolean")
    strips = options.get("strip_prefixes", ())
    if not isinstance(strips, (list, tuple)) or any(
        not isinstance(value, str) for value in strips
    ):
        raise ValueError("initialization.strip_prefixes must be a list of strings")
    renames = options.get("rename_prefixes")
    if renames is not None and (
        not isinstance(renames, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in renames.items()
        )
    ):
        raise ValueError(
            "initialization.rename_prefixes must map string prefixes to strings"
        )
    if (strips or renames) and options.get("strict", True) is False:
        raise ValueError("initialization key conversion requires strict=True")
    return options


def _read_state(path: str | Path) -> tuple[Mapping, str | None, int | None]:
    """Decode only the raw/state_dict/model containers already supported."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("Model weights must be a state_dict or checkpoint mapping")
    for container in ("state_dict", "model"):
        if isinstance(payload.get(container), Mapping):
            return payload[container], container, payload.get("global_step")
    return payload, None, None


def _convert_keys(state, strip_prefixes, rename_prefixes):
    """Apply declared conversions once, rejecting ambiguous target keys."""
    if isinstance(strip_prefixes, (str, bytes)) or not isinstance(
        strip_prefixes, Sequence
    ):
        raise ValueError("strip_prefixes must be a sequence of strings")
    if any(not isinstance(prefix, str) for prefix in strip_prefixes):
        raise ValueError("strip_prefixes must contain strings")
    if rename_prefixes is not None and (
        not isinstance(rename_prefixes, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in rename_prefixes.items()
        )
    ):
        raise ValueError("rename_prefixes must map string prefixes to strings")
    if not strip_prefixes and not rename_prefixes:
        # Native state_dict mappings carry per-module version metadata.
        return state
    converted = OrderedDict()
    for key, value in state.items():
        for prefix in strip_prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        for old, new in (rename_prefixes or {}).items():
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        if key in converted:
            raise ValueError(f"Checkpoint key conversion collides at {key}")
        converted[key] = value
    if hasattr(state, "_metadata"):
        # Metadata names modules without a trailing dot. Native prefix
        # consumption assumes a single wrapper name for its root special
        # case, so nested module selection needs an exact path comparison.
        metadata = deepcopy(state._metadata)
        for prefix in strip_prefixes:
            root_name = prefix.removesuffix(".")
            retained, moved = OrderedDict(), OrderedDict()
            for name, local_metadata in metadata.items():
                if name and (name == root_name or name.startswith(prefix)):
                    target = "" if name == root_name else name[len(prefix) :]
                    moved[target] = local_metadata
                else:
                    retained[name] = local_metadata
            # Selected modules replace wrapper metadata regardless of source
            # ordering; moving into a source key must not consume it twice.
            retained.update(moved)
            metadata = retained
        converted._metadata = OrderedDict()
        for name, local_metadata in metadata.items():
            qualified_name = name + "." if name else ""
            for old, new in (rename_prefixes or {}).items():
                if qualified_name.startswith(old):
                    qualified_name = new + qualified_name[len(old) :]
                    break
            target = qualified_name.removesuffix(".")
            if target and target in converted._metadata:
                raise ValueError(f"Checkpoint metadata conversion collides at {target}")
            converted._metadata[target] = local_metadata
    return converted


def restore_model(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool = True,
    strip_prefixes: Sequence[str] = (),
    rename_prefixes: Mapping[str, str] | None = None,
) -> int | None:
    """Restore model-only weights without optimizer or training-loop state.

    Raw and ``model`` containers retain their keys. A MetricModule ``state_dict``
    selects only ``model.`` keys when present, excluding the criterion. A
    ``state_dict`` without that namespace is a plain model state. Explicit
    conversions are applied afterwards. Return a stored container training step,
    or None for raw model weights.
    """
    state, container, training_step = _read_state(checkpoint)
    if container == "state_dict":
        model_state = OrderedDict(
            (name, value) for name, value in state.items() if name.startswith("model.")
        )
        metadata = getattr(state, "_metadata", {})
        if model_state or "model" in metadata:
            if hasattr(state, "_metadata"):
                # Exclude wrapper/criterion versions; the target model root
                # receives the former "model" metadata, not the wrapper's.
                model_state._metadata = OrderedDict(
                    (name, deepcopy(metadata))
                    for name, metadata in state._metadata.items()
                    if name == "model" or name.startswith("model.")
                )
            consume_prefix_in_state_dict_if_present(model_state, "model.")
            state = model_state
    state = _convert_keys(state, strip_prefixes, rename_prefixes)
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        logging.getLogger(__name__).warning(
            "Partial initialization: missing=%s unexpected=%s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
    return training_step


def load_pretrained(
    model: nn.Module,
    path: str | Path,
    *,
    strip_prefixes: Sequence[str] = (),
    rename_prefixes: Mapping[str, str] | None = None,
) -> None:
    """Strictly load raw/state_dict/model weights using declared key conversions.

    Strip each prefix once, in order; then apply the first matching rename.
    No method-name heuristics, silent missing keys, or duplicate target keys.
    """
    state, _, _ = _read_state(path)
    model.load_state_dict(
        _convert_keys(state, strip_prefixes, rename_prefixes), strict=True
    )
