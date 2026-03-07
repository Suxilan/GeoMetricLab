# Models Module

This module contains the core model implementations behind GeoMetricLab.

## Scope

- `GeoEncoder` as the main backbone-plus-aggregator abstraction
- Backbone wrappers for CNN and ViT-based feature extraction
- Aggregators that convert local features or tokens into compact global descriptors
- Supporting modules required by the public model stack

## Architecture overview

GeoMetricLab models are organized around a simple composition pattern:

1. A backbone extracts spatial or token-level representations from an image
2. An aggregator turns those representations into a global descriptor
3. The top-level encoder normalizes and exposes a consistent retrieval embedding interface

## Main components

- `geoencoder.py`: public model entrypoint used by training and evaluation scripts
- `backbone/`: CNN and transformer backbone wrappers
- `aggregator/`: maintained retrieval aggregation heads
- `modules/`: shared internal layers used by the model stack

## Maintained public aggregators

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

## Notes

- Third-party method provenance is recorded in the corresponding source files
- This README intentionally focuses on maintained public model surfaces# GeoEncoder Model Architecture

This directory contains the implementation of the **GeoEncoder**, a modular feature extraction framework designed for Visual Geo-localization and Image Retrieval tasks.

## Architecture Overview

The GeoEncoder consists of two main components:
1.  **Backbone**: Extracts local features or token representations from the input image.
2.  **Aggregator**: Aggregates the backbone outputs into a global feature vector.

### 1. Backbone (`models/backbone/`)

The backbone is responsible for processing the raw image. We support two types of backbones with distinct output formats:

*   **CNN-based (ResNet, VGG, etc.)**:
    *   **Output**: A single tensor representing the feature map `(B, C, H, W)`.
    *   **Design**: Classification heads (FC layers) are removed.
*   **ViT-based (DINOv2, DINOv3, etc.)**:
    *   **Output**: A tuple `(cls_token, patch_tokens, attn_maps)`.
        *   `cls_token`: `(B, D)`
        *   `patch_tokens`: `(B, D, H, W)` (reshaped from `(B, N, D)`)
        *   `attn_maps`: `(B, H, N, N)` or `None`.

### 2. Aggregator (`models/aggregator/`)

The aggregator takes the backbone output and produces a single global feature vector.

*   **Supported Aggregators**:
    *   `Avg`: Global Average Pooling (GAP).
    *   `GeM`: Generalized Mean Pooling.
    *   `CosPlace`: CosPlace aggregation (from `third_party`).
    *   `Salad`: SALAD aggregation (from `third_party`).
    *   `BoQ`: Bag of Queries (from `third_party`).
    *   `CLS`: Selects the CLS token (specific to ViT backbones).
    *   *(Future)*: `RMAC`, `NetVLAD`, etc.

### 3. GeoEncoder (`models/geoencoder.py`)

The top-level module that combines a backbone and an aggregator.

*   **Features**:
    *   Modular instantiation via config/names.
    *   Handles the dispatch of backbone outputs to aggregators (e.g., passing `patch_tokens` to GeM, or `cls_token` to CLS aggregator).
    *   **L2 Normalization**: The final global feature is always L2 normalized.
    *   **Weights Loading**: Supports loading pre-trained weights from various SOTA methods, handling key remapping.

## Usage

```python
from models.geoencoder import GeoEncoder

# Example 1: ResNet50 + GeM
model = GeoEncoder(
    backbone="resnet50",
    aggregator="gem",
    backbone_args={"pretrained": True, "crop_last_block": True},
    aggregator_args={"p": 3.0}
)

# Example 2: DINOv2 + CLS
model = GeoEncoder(
    backbone="dinov2_vitb14",
    aggregator="cls",
    backbone_args={"num_layers": 12}
)

x = torch.randn(1, 3, 224, 224)
global_feat = model(x) # (1, D)
```
