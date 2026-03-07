# Pipeline Module

This module contains the training framework abstractions that connect datasets, models, optimization, and logging.

## Scope

- Lightning-style DataModule and Framework orchestration
- Instance retrieval training pipeline
- Supervised scene training pipeline

## Public focus

The documented and tracked training flows in this repository are the instance and supscene pipelines used by the public training entrypoints.

## Structure

- `instance/`: instance-level retrieval training components
- `supscene/`: supervised scene training components
- `__init__.py`: exported pipeline symbols