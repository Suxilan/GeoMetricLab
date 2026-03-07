# Rerank Module

This module contains post-retrieval reranking utilities for descriptor refinement and nearest-neighbor reordering.

## Scope

- FAISS-based nearest-neighbor search helpers
- Query expansion and database augmentation style rerankers
- Graph-based and reciprocal-neighbor reranking utilities

## Notes

- These utilities are designed as optional post-processing components
- They are kept separate from the main `GeoEncoder` model definition to preserve modularity