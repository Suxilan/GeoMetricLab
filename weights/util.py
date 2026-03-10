"""Pretrained weight loader utilities (responsibility: pretrained weights only)."""

from __future__ import annotations

import torch
from src.utils.logger import print_rank_0
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from copy import deepcopy

# Loadable weight configurations. For readability and maintainability we
# prefer an external YAML file: `weights/weights_config.yml`.
_WEIGHT_CONFIGS: Dict[str, Dict[str, Any]]


def _load_weight_configs() -> Dict[str, Dict[str, Any]]:
    """Attempt to load weight configs from `weights/weights_config.yml`.

    If PyYAML is not installed or the YAML file is absent, fall back to the
    embedded defaults below to preserve backward compatibility.
    """
    import json

    cfg_path = Path("weights") / "weights_config.yml"

    # Try YAML first if available
    try:
        import yaml  # type: ignore

        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
                if isinstance(data, dict):
                    print_rank_0(f"Loaded weight configs from {cfg_path}.")
                    return data
    except Exception:
        # Missing PyYAML or parse error — we'll try JSON or fallback
        pass

    # Try JSON file as alternative (weights/weights_config.json)
    json_path = Path("weights") / "weights_config.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    print_rank_0(f"Loaded weight configs from {json_path}.")
                    return data
        except Exception:
            pass

    # Strict mode: require an external config file (no embedded fallback).
    raise FileNotFoundError(
        "weights/weights_config.yml not found. Please create the file at 'weights/weights_config.yml' "
        "or provide 'weights/weights_config.json'. Install PyYAML to use YAML: `pip install pyyaml`."
    )


_WEIGHT_CONFIGS = _load_weight_configs()

# Method-specific key remapping registry (normalized by stem, lower-case).
_METHOD_REMAPS: Dict[str, list[Callable[[str], str]]] = {}


def _resolve_weight_entry(weight_arg: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Find matching weight entry and resolved path (if the file exists)."""
    weight_path: Optional[Path] = None
    candidates = []

    p = Path(weight_arg)
    if p.exists():
        weight_path = p
        candidates.extend([p.name, p.stem])
    else:
        pw = Path("weights") / weight_arg
        if pw.exists():
            weight_path = pw
            candidates.extend([pw.name, pw.stem])

    candidates.append(weight_arg)

    entry = None
    for key in candidates:
        if key in _WEIGHT_CONFIGS:
            entry = _WEIGHT_CONFIGS[key]
            break

    return weight_path, entry


def get_weight_config(weight_arg: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    """Resolve weight path and model config for a given pretrained weight key/path."""
    weight_path, entry = _resolve_weight_entry(weight_arg)

    if entry is None:
        print_rank_0(f"Error: No config found for {weight_arg}. Available: {list_available_weights()}")
        return None, None

    model_cfg = deepcopy(entry["model_cfg"])

    if weight_path is None and entry.get("weight_path"):
        weight_path = Path(entry["weight_path"])

    return weight_path, model_cfg

def list_available_weights() -> list:
    """List all available weight configurations."""
    return list(_WEIGHT_CONFIGS.keys())


def _strip_wrappers(key: str) -> str:
    """Remove common training-time prefixes such as module./model."""
    if key.startswith("module."):
        key = key[len("module."):]
    if key.startswith("model."):
        key = key[len("model."):]
    return key


def _normalize_tag(name: str | Path) -> str:
    """Normalize a weight identifier/path to a lower-case stem for matching."""
    return Path(str(name)).stem.lower()


def _remap_dino_backbone(key: str) -> str:
    """Handle Dino checkpoints that store the vision transformer under model."""
    if key.startswith("backbone.model."):
        return "backbone.dino." + key[len("backbone.model."):]
    if key.startswith("model."):
        return "backbone.dino." + key[len("model."):]
    return key


def _remap_resnet_backbone(key: str) -> str:
    """Some ResNet checkpoints omit the Sequential name 'net' inside backbone."""
    if key.startswith("backbone."):
        suffix = key[len("backbone."):]
        if not suffix.startswith("net."):
            return "backbone.net." + suffix
    return key


def _remap_cosplace_legacy(key: str) -> str:
    """Legacy CosPlace checkpoints use aggregation.* names for the head."""
    if key.startswith("aggregation.1.p"):
        return "aggregator.gem.p"
    if key == "aggregation.3.weight":
        return "aggregator.fc.weight"
    if key == "aggregation.3.bias":
        return "aggregator.fc.bias"
    return key


# Register built-in remappers for shipped weights (extend here for new weights).
_METHOD_REMAPS.update(
    {
        "resnet50_512_cosplace": [_remap_resnet_backbone, _remap_cosplace_legacy],
        "resnet50_boq_16384": [_remap_resnet_backbone],
        "resnet50_gem": [_remap_resnet_backbone],
        "dino_salad": [_remap_dino_backbone],
        "dinov2_boq_12288": [_remap_dino_backbone],
    }
)


def _remap_key(key: str, method_name: str, weight_path: Path | str | None = None) -> str:
    """Map checkpoint keys to GeoEncoder naming conservatively and per method.

    Remapping is now opt-in per method to keep future extensions safe: only
    normalized weight names registered in `_METHOD_REMAPS` (or those that match
    simple substrings like "dino"/"cosplace") get extra rules. Unknown weights
    only have lightweight wrapper-stripping applied.
    """

    key = _strip_wrappers(key)

    # Collect hints from method name and weight filename
    hints = {_normalize_tag(method_name)}
    if weight_path is not None:
        hints.add(_normalize_tag(weight_path))

    # Build remap pipeline (method-specific first, then heuristic fallbacks)
    pipeline: list[Callable[[str], str]] = []
    for hint in hints:
        pipeline.extend(_METHOD_REMAPS.get(hint, []))

    # Heuristics to cover unregistered but obvious cases
    joined_hints = " ".join(hints)
    if not any(fn is _remap_dino_backbone for fn in pipeline) and "dino" in joined_hints:
        pipeline.append(_remap_dino_backbone)
    if not any(fn is _remap_cosplace_legacy for fn in pipeline) and "cosplace" in joined_hints:
        pipeline.append(_remap_cosplace_legacy)
    if not any(fn is _remap_resnet_backbone for fn in pipeline) and "resnet" in joined_hints:
        pipeline.append(_remap_resnet_backbone)

    # Deduplicate while preserving order
    seen: set[Callable[[str], str]] = set()
    ordered_pipeline = []
    for fn in pipeline:
        if fn not in seen:
            ordered_pipeline.append(fn)
            seen.add(fn)

    for fn in ordered_pipeline:
        new_key = fn(key)
        if new_key != key:
            key = new_key

    return key


def load_weights(model, weight_path: Path | str | None, method_name: str):
    """Load pre-trained weights into the GeoEncoder with basic key remapping.

    Args:
        model: GeoEncoder instance.
        weight_path: Path to weight file (can be Path or str).
        method_name: Logical weight key (e.g., 'ResNet50_512_cosplace').
    """
    if not weight_path:
        raise FileNotFoundError(f"No weight path provided for '{method_name}'.")

    print_rank_0(f"Loading weights from {weight_path} for {method_name}...")

    try:
        checkpoint = torch.load(weight_path, map_location="cpu")
    except Exception as exc:
        print_rank_0(f"Failed to load checkpoint: {exc}")
        return [], []

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = _remap_key(k, method_name, weight_path)
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print_rank_0(f"Weights loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if missing:
        print_rank_0(f"Missing keys: {missing}")
    if unexpected:
        print_rank_0(f"Unexpected keys: {unexpected}")
    print_rank_0("Weight loading complete.")
    return missing, unexpected
