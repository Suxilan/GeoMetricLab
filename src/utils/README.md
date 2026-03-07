# Utils Module

This module contains shared infrastructure utilities used across training, evaluation, and analysis.

## Scope

- Logging and callback helpers
- Metric IO and descriptor serialization
- Visualization helpers for local inspection

## Key files

- `metrics.py`: retrieval metric helpers
- `io.py`: cache and feature IO utilities
- `logger.py`: logging support
- `callbacks.py`: reusable training callbacks

## Notes

- Visualization scripts in this module are developer-oriented helpers
- Large generated figures should stay outside version control