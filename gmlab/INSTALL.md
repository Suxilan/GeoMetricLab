# Installation

The import package is `gmlab`; the distribution name is `geometriclab`.
Run source-install commands from the repository root. Installation applies to
the selected Python environment and makes the library usable from any directory.
Use Python 3.12 for the reference environment.

## Conda environment

The root `environment.yml` declares Python, NumPy, GPU FAISS and its C++ runtime.
Its pip section uses the root `requirements.txt` to install the maintained
workflow extras and the library in editable mode.

```bash
make conda-env
conda activate GeoMetricLab
make check-env
```

The reference environment targets Linux/NVIDIA. Its FAISS CUDA 13.2 build needs
a compatible NVIDIA driver. `make conda-env ENV_NAME=my-gmlab` creates a separate
named environment. Do not use this target to recreate an existing environment.
The equivalent manual sequence is:

```bash
conda env create -f environment.yml
conda run -n GeoMetricLab python tools/environment/configure_conda.py
conda activate GeoMetricLab
```

The configuration command installs reversible activation/deactivation hooks into
that environment. These make Torch and FAISS use Conda's C++ runtime regardless
of import order. Merely creating the YAML environment does not install the hooks.

## pip and source installation

Within an existing virtual/Conda environment:

```bash
python -m pip install -r requirements.txt
```

This installs editable sources and the runtime, models, peft, artifacts and
swanlab extras. Dependency ranges are maintained once, in `pyproject.toml`;
`requirements.txt` is not a freeze of unrelated workstation packages.
For a regular installation with the same extras, use `make install`.
For editable installation plus Conda runtime configuration, use `make develop`.
`PYTHON=/path/to/python` selects an explicit interpreter for either Make target.

For just the base library:

```bash
python -m pip install .
# Or, while developing the library:
python -m pip install -e .
```

`-e` keeps imports linked to this checkout, so code edits take effect immediately.
A regular installation copies the package into the environment and must be
reinstalled after changes. Neither method requires preparing a wheel manually,
setting PYTHONPATH, or running from the repository afterward. Separate Python
environments need separate installations.

The pip-only route supports Torch retrieval. GPU FAISS is provided by the Conda
recipe; do not install a pip FAISS package over that Conda package. A separate
pip-only environment can optionally install `faiss-cpu` for CPU FAISS consumers.

## Repair and verify an existing environment

Activate the intended Conda environment, then run:

```bash
make repair-env
conda deactivate
conda activate GeoMetricLab
make check-env
cd /tmp
python -c "import gmlab; print(gmlab.__file__)"
python -m gmlab.train --help
```

The hooks prioritize that environment's `lib/` and restore the previous
LD_LIBRARY_PATH when deactivated. They do not modify system libraries or shell
startup files. Existing Python processes must be restarted. IDEs or services that
launch the interpreter without Conda activation must use equivalent environment
variables; choosing the interpreter path alone does not execute activation hooks.
The check imports Torch before FAISS and runs a small CPU/GPU search comparison
when FAISS and a GPU are available. This is not a performance benchmark.

Model repositories, checkpoints and datasets remain explicit inputs. Relative
recipe paths are resolved from the working directory; use absolute paths when
launching from another directory.

## Optional dependency groups

Choose only the extras used by the experiment:

| Dependency group | Used for |
| --- | --- |
| Base: torch, torchvision, PML, NumPy, Pillow | Injected models, metric losses, common data tools and Torch retrieval |
| runtime | Lightning training/evaluation CLI, YAML and Rich |
| models | Built-in timm ResNet construction |
| peft | DINOv2 LoRA injection and the local DINOv2 model's einops dependency |
| swanlab | Official SwanLab logger; logger=False does not require this group |
| artifacts | H5 feature readers/writers; extraction also requires runtime for Fabric container utilities |

PEFT DINOv2 also needs the local DINOv2 repository configured by
model.backbone.repository. It is an explicit external model implementation, not
bundled or downloaded implicitly by gmlab. Pretrained DINO weights use PyTorch's
normal cache/download path. Cache availability is required for offline runs.

## Installed resources

The installed library contains `gmlab` and shared dataset utilities. YAML recipes
are available under `<prefix>/share/geometriclab/recipes`. Concrete readers live
in the source checkout's `datasets/` directory and are not installed as a global
`datasets` package. Run the supplied dataset examples from the repository root;
for another project, use its installed reader's Python path in the recipe.
The source archive includes these examples and the `tools/` and `scripts/`
directories. Dataset assets, weights, results and legacy code are excluded.

Use `python -m gmlab.train --help`, `python -m gmlab.evaluate --help` and
`python -m gmlab.artifacts --help` with their corresponding extras installed.
Feature extraction requires both `artifacts` and `runtime`; H5 readers alone need
`artifacts`.

## External assets and limitations

Provide dataset paths, GT, model checkpoints and external model repositories
explicitly. AnyLoc vocabularies and native COLMAP executables are separate assets.
Installation does not generate vocabularies, fit whitening or launch experiments.
Base dependencies do not include every checkout tool's optional dependencies.

Dependency ranges in `pyproject.toml` are metadata constraints, not a guarantee
that every Python/OS/CUDA combination has been exercised. Feature publication
uses local POSIX locks; shared NFS and multi-node publication are not supported.
Checkpoint restore does not guarantee arbitrary data-loader cursor recovery.
