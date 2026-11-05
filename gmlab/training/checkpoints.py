"""Model initialization and explicit selection of saved training checkpoints."""

import json
from collections.abc import Mapping
from pathlib import Path

from torch import nn

from ..models.weights import restore_model, validate_initialization


def initialize_model(model: nn.Module, initialization: Mapping | None) -> None:
    """Apply model-only initialization; full training resume remains Lightning's.

    Shared weight IO handles raw/state_dict/model containers and explicit prefix
    conversions. Independent evaluation loads its selected checkpoint directly.
    """
    options = validate_initialization(initialization)
    checkpoint = options.pop("checkpoint", None)
    if checkpoint is not None:
        restore_model(model, checkpoint, **options)


def resolve_checkpoint(
    checkpoint: str | Path | None = None,
    run_dir: str | Path | None = None,
    weights: str | None = None,
) -> Path:
    """Select an explicit file or the best/last checkpoint recorded by one run."""
    if checkpoint is not None:
        if run_dir is not None or weights is not None:
            raise ValueError("Choose checkpoint or run_dir + best/last")
        return Path(checkpoint)
    if run_dir is None or weights not in {"best", "last"}:
        raise ValueError("Checkpoint selection requires run_dir and best/last")

    manifest = json.loads((Path(run_dir) / "run.json").read_text())
    key = "best_checkpoint" if weights == "best" else "checkpoint"
    selected = manifest.get(key)
    if not selected:
        raise ValueError(f"No {weights} checkpoint in run manifest")
    return Path(selected)
