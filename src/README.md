# Source Tree Overview

This directory contains the core implementation of GeoMetricLab.

## Top-level modules

- `datasets/`: dataset adapters and data-loading components
- `eval/`: reusable benchmark and evaluation utilities
- `losses/`: metric-learning losses and related helpers
- `models/`: GeoEncoder, backbones, and aggregators
- `pipeline/`: public training pipeline abstractions
- `rerank/`: optional post-retrieval refinement utilities
- `supscene/`: supervised-scene task components
- `utils/`: shared infrastructure and visualization helpers

Each module has its own local `README.md` describing its role in the repository.