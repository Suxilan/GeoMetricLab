<p align="center">
	<img src="assets/GeoMetricLab.png" alt="GeoMetricLab banner" width="100%" />
</p>

<h1 align="center">GeoMetricLab-Geometry/Geography Metric Learning</h1>

<p align="center">
	<a href="https://www.python.org/">
		<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
	</a>
	<a href="https://pytorch.org/">
		<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
	</a>
	<a href="https://lightning.ai/">
		<img src="https://img.shields.io/badge/Lightning-Training%20Framework-792EE5?logo=lightning&logoColor=white" alt="Lightning" />
	</a>
	<a href="https://github.com/facebookresearch/dinov2">
		<img src="https://img.shields.io/badge/VFM-DINOv2%20%7C%20DINOv3-0A66C2" alt="Vision Foundation Models" />
	</a>
		<a href="https://github.com/KevinMusgrave/pytorch-metric-learning">
		<img src="https://img.shields.io/badge/PyTorch%20Metric%20Learning-PML-2EA44F?logo=github" alt="PyTorch Metric Learning" />
	</a>
</p>

<p align="center"><strong>Unified Geo-Localization, VPR, and Retrieval Research Toolbox</strong></p>

<p align="center">Composable GeoEncoder pipelines for backbone exploration, descriptor aggregation, metric learning, evaluation, and reproducible geometric retrieval experiments.</p>

GeoMetricLab is a research-oriented toolbox for visual geo-localization, visual place recognition (VPR), and image retrieval. It centers on a unified `GeoEncoder` abstraction that composes modern CNN / ViT backbones with retrieval-oriented aggregation heads, making it easy to train, evaluate, and compare geometry-aware global descriptors in a single codebase.

## ✨ Highlights

- Unified encoder design for backbone + aggregator composition
- Support for both CNN and transformer-style feature pipelines
- Ready-to-run training engines for GL3D and University-1652 workflows
- Lightweight evaluation scripts with optional feature-cache loading
- Clean model hubs for canonical presets, transforms, losses, and datasets
- Third-party methods integrated with explicit provenance and submodule management

## 🧭 What this repository focuses on

GeoMetricLab targets the representation-learning layer of geo-localization and retrieval pipelines:

1. Extract dense or token features with a configurable backbone
2. Aggregate local descriptors into a compact global representation
3. Optionally apply BN, whitening, and final normalization
4. Train with task-specific pipelines and evaluate with retrieval metrics

The repository is designed for fast iteration on:

- backbone selection
- aggregation design
- descriptor dimensionality
- metric-learning objectives and schedules
- cached-feature evaluation and deployment-style inference

## 🧱 Core model stack

### 🔹 Backbones

Current backbone registry includes:

- ResNet / ResNeXt
- DINOv2
- PEFT-DINOv2
- DINOv3
- Swin Transformer V2
- ConvNeXt

### 🔹 Aggregators

The maintained aggregator set in this repository includes:

- Avg
- CLS
- GeM
- BoQ
- CosPlace
- EigenPlace
- NetVLAD
- GhostVLAD
- SuperVLAD
- SALAD
- CricaVPR

Several implementations are adapted from their original public repositories, with per-file attribution placed near the corresponding class or function definitions.

## ⚙️ Supported capabilities

GeoMetricLab currently supports the following workflows:

- training instance-level retrieval models
- training supervised scene-level retrieval models
- evaluating GL3D and University-1652 retrieval pipelines
- loading descriptors from cache for fast offline benchmarking
- exporting or reusing trained encoder weights
- experimenting with LoRA-style PEFT adaptation on DINOv2 backbones
- initializing VLAD-style aggregators with FAISS-based clustering utilities

## 🏗️ Training pipelines

The main training entrypoints live in `engine/`:

- `engine/train_instance_gl3d_engine.py`
- `engine/train_instance_u1652_engine.py`
- `engine/train_supscene_gl3d_engine.py`

The reusable data / framework components in `src/pipeline/` currently cover:

- instance metric learning
- supervised scene training

## 📏 Evaluation entrypoints

Tracked evaluation scripts live in `scripts/`:

- `scripts/eval_gl3d.py`
- `scripts/eval_university1652.py`

These scripts support two common evaluation modes:

- loading pre-extracted descriptors from cache
- building a `GeoEncoder` and extracting features from weights directly

## 🗂️ Dataset coverage

### 🔹 Implemented training / validation datasets

- GL3D(BlendedMVS)
- University-1652

### 🔹 Repository data layout also includes

- ROxford
- RParis
- SfM-120k
- GSVCities
- MSLS

Some of these datasets are used as local evaluation resources or experiment assets, while the officially maintained training engines in this repository currently target GL3D and University-1652 first.

## 🧪 Tech stack

GeoMetricLab is built around a practical research stack for modern retrieval experiments:

- PyTorch for model definition and tensor computation
- PyTorch Lightning / Lightning for training orchestration
- torchvision for transforms and model utilities
- PyTorch Metric Learning for retrieval losses, miners, and distance functions
- FAISS for fast nearest-neighbor search and VLAD-related clustering utilities
- h5py for descriptor cache storage and feature IO
- Weights & Biases and TensorBoard for experiment logging
- PEFT for LoRA-style efficient adaptation of Visual Foundation Model backbones

## 🔌 Third-party ecosystem

External projects under `third_party/` are tracked as submodules rather than vendored into the main repository. The current repository layout includes method ecosystems such as:

- DINOv2
- DINOv3
- BoQ
- CosPlace
- CricaVPR
- SALAD

This keeps method provenance explicit while making it easier to update or compare upstream implementations.

## 🗃️ Project structure

```text
GeoMetricLab/
├── config/              # canonical model / transform / loss configs
├── data/                # dataset roots (often local symlinks or external mounts)
├── engine/              # train entrypoints
├── scripts/             # tracked evaluation entrypoints
├── src/
│   ├── datasets/        # dataset implementations
│   ├── models/          # backbones, aggregators, encoder modules
│   ├── pipeline/        # datamodules + training frameworks
│   └── utils/           # metrics, IO, logging, callbacks
├── third_party/         # external dependencies as submodules
├── weights/             # local model weights
└── notebooks/           # analysis and visualization notebooks
```

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Suxilan/GeoMetricLab.git
cd GeoMetricLab
```

### 2. Initialize submodules

```bash
git submodule update --init --recursive
```

### 3. Install dependencies

```bash
pip install -e .
```

Install your PyTorch stack separately if you need a CUDA-specific build.

## 🧰 Data preparation

This repository expects datasets to be placed under `data/`. In practice, large datasets are often mounted or linked from external storage.

Typical strategy:

- keep raw datasets outside the repo
- create symlinks into `data/`
- keep experiment caches and checkpoints local

Refer to `data/README.md` for dataset-specific layout notes.

## ▶️ Quick start

### Train on GL3D instance learning

```bash
python engine/train_instance_gl3d_engine.py \
	--config config/train_instance_gl3d/default.yaml
```

### Train on GL3D SupScene

```bash
python engine/train_supscene_gl3d_engine.py \
	--config config/train_supscene_gl3d/default.yaml
```

### Train on University-1652 instance learning

```bash
python engine/train_instance_u1652_engine.py \
	--config config/train_instance_u1652/default.yaml
```

### Evaluate on GL3D

```bash
python scripts/eval_gl3d.py \
	--model dino_salad \
	--root ./data/GL3D/test
```

### Evaluate on University-1652

```bash
python scripts/eval_university1652.py \
	--model resnet50_cosplace \
	--task u1652_drone2satellite \
	--root ./data/university1652/test
```

## 🧩 Model hubs

Canonical model presets are defined in `config/model_hubs.py`. These entries provide a stable way to reference common backbone / aggregator combinations such as:

- `resnet50_gem`
- `resnet50_cosplace`
- `resnet50_boq_16384`
- `dinov2_boq_12288`
- `dino_salad`

This makes it easier to keep evaluation and weight loading consistent across experiments.

## 🎯 Design principles

- Keep the encoder modular and easy to recompose
- Separate experimental weights and caches from source code
- Preserve third-party method provenance explicitly
- Prefer simple training and evaluation entrypoints over deeply coupled runners

## 📌 Status

GeoMetricLab is an active research codebase. The repository prioritizes reproducible experimentation, modular method integration, and fast empirical iteration over packaging polish. Interfaces may evolve as new backbones, aggregators, and benchmarks are added.

## 🙏 Acknowledgements

GeoMetricLab is built on top of a strong open-source ecosystem in visual retrieval, VPR, and modern vision foundation models. Special thanks to the communities and projects behind:

- [OpenVPRLab](https://github.com/openvprlab), for helping shape open and reproducible VPR research practices
- PyTorch and PyTorch Lightning, for the core training and experimentation framework
- [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning), for robust retrieval-oriented losses, miners, and metric utilities
- FAISS, for efficient large-scale nearest-neighbor search and clustering primitives
- [DINOv2](https://github.com/facebookresearch/dinov2) and [DINOv3](https://github.com/facebookresearch/dinov3), for strong vision foundation backbones
- public method repositories such as [SALAD](https://github.com/serizba/salad), [CosPlace](https://github.com/gmberton/CosPlace), [BoQ](https://github.com/amaralibey/Bag-of-Queries), and [CricaVPR](https://github.com/Lu-Feng/CricaVPR), which make comparative retrieval research much easier

## 📚 Citation

If GeoMetricLab contributes to your research workflow, please cite the original papers of the backbone and aggregation methods you use.
