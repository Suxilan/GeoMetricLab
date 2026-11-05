"""Launch one YAML metric-learning experiment: python -m gmlab.train --config ..."""

import argparse
import copy
import os
import tempfile
from pathlib import Path

import pytorch_lightning as pl
import yaml

from .components import build_component
from .config import read_training_recipe, validate_training_config
from .training.callbacks import RNGState
from .training.recipes import build_training_components
from .training.run_log import RunLogger
from .training.runtime import fit
from .utils.banner import show_startup_banner


def development_config(config: dict, batches: int) -> dict:
    """Run native Lightning fast_dev_run with the selected model/data/precision.

    Full retrieval metrics need complete collections and are skipped in this mode.
    Tracking, checkpoint selection and early stopping are disabled by construction.
    The source recipe is left intact; only the execution policy changes.
    """
    if type(batches) is not int or batches < 1:
        raise ValueError("--dev requires a positive batch count")
    config = copy.deepcopy(config)
    runtime = config["runtime"]
    runtime["trainer"]["fast_dev_run"] = batches
    runtime["run_dir"] = str(Path(runtime["run_dir"]) / "dev")
    runtime["final_validation"] = "scheduled"
    runtime["validate_before_fit"] = False
    runtime["test_weights"] = "current"
    runtime["checkpoint"] = {"monitor": None, "save_top_k": 0, "save_last": False}
    runtime["save_rng"] = False
    runtime["logging"].pop("tracking", None)
    runtime["logging"]["logger"] = False
    return config


def _development_directory(root: str) -> str:
    """Share one fresh output directory with Lightning's re-executed DDP ranks."""
    inherited = os.environ.get("GMLAB_DEV_RUN_DIRECTORY")
    if inherited is not None and "LOCAL_RANK" in os.environ:
        return inherited
    Path(root).mkdir(parents=True, exist_ok=True)
    directory = tempfile.mkdtemp(prefix="run-", dir=root)
    os.environ["GMLAB_DEV_RUN_DIRECTORY"] = directory
    return directory


def main() -> None:
    """Resolve one YAML experiment, construct its components and call public fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Restore a complete training checkpoint")
    parser.add_argument(
        "--dev",
        nargs="?",
        const=1,
        type=int,
        metavar="BATCHES",
        help="Smoke-test this recipe with 1 (or BATCHES) batches per loop; no metrics/checkpoints",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Validate and print resolved YAML without loading data or weights",
    )
    args = parser.parse_args()
    config = read_training_recipe(
        args.config, resume=args.resume, validate=args.dev is None
    )
    if args.dev is not None:
        config = development_config(config, args.dev)
        validate_training_config(config)
    runtime = config["runtime"]
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    if args.dev is not None:
        runtime["run_dir"] = _development_directory(runtime["run_dir"])
    run_dir = Path(runtime["run_dir"])
    if (run_dir / "run.json").exists() and not runtime.get("resume"):
        raise FileExistsError(
            f"Run already exists: {run_dir}; choose a new directory or resume"
        )
    with RunLogger(
        run_dir / "train.log", runtime["logging"].get("text_level", "INFO")
    ) as run_log:
        show_startup_banner(run_log.console)
        run_log.environment()
        pl.seed_everything(runtime.get("seed", 81), workers=True)
        components = build_training_components(config)
        callbacks = [build_component(value) for value in runtime.get("callbacks", [])]
        if runtime.get("save_rng", False):
            callbacks.append(RNGState())
        fit(
            components.module,
            components.train_loader,
            components.valid_loader,
            test_loader=components.test_loader,
            ckpt_path=runtime.get("resume"),
            run_dir=runtime["run_dir"],
            trainer_options=runtime["trainer"],
            checkpoint_options=runtime["checkpoint"],
            final_validation=runtime.get("final_validation", "scheduled"),
            validate_before_fit=runtime.get("validate_before_fit", False),
            test_weights=runtime.get("test_weights", "current"),
            callbacks=callbacks,
            metadata={
                "resolved_config": config,
                "data_metadata": components.valid_metadata,
            },
            **runtime["logging"],
        )


if __name__ == "__main__":
    main()
