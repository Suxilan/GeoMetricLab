# Datasets Module

This module contains dataset adapters and data-loading utilities used by GeoMetricLab training and evaluation pipelines.

## Scope

- Training and validation datasets for retrieval experiments
- Dataset-specific collate functions and sampling helpers
- Shared dataset interfaces used by the tracked engines and evaluation scripts

## Public focus

This repository currently documents and maintains the dataset flows that support the public GL3D and University-1652 pipelines.

## Structure

- `train/`: dataset definitions used during optimization
- `valid/`: validation and benchmark-oriented dataset wrappers
- `__init__.py`: exported dataset registry surface

## Notes

- Large raw datasets are expected to live outside the repository and be linked into `data/`
- Dataset layout details belong in project-level data documentation rather than source code comments