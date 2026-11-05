# Tools

Run these commands from the repository root. Each has `--help`.
For complete workflows, see [scripts](../scripts/README.md).

| Tool | Input → output |
| --- | --- |
| `environment/check.py` | Active environment → dependency and FAISS checks |
| `environment/configure_conda.py` | Conda environment → runtime activation hooks |
| `features/extract_image_features.py` | Images + model recipe → feature records and H5 |
| `retrieval/retrieve.py` | Query/gallery feature records → ranked JSONL |
| `retrieval/build_retrieval_pairs.py` | Feature record → COLMAP image pairs |
| `retrieval/evaluate_retrieval_pairs.py` | Pairs + GT or geometric matches → evaluation |
| `retrieval/classical_retrieval.py` | Images → SIFT/BoW retrieval |
| `weights/assemble_whitening_model.py` | Compatible feature records → whitening fit and model assembly |
| `features/build_vlad_kmeans.py` | Spatial features → VLAD vocabulary |
| `python -m gmlab.artifacts` | Cache → inventory or integrity report |

## Extract features

Set model weights and preprocessing in the recipe, then run:

```bash
python tools/features/extract_image_features.py \
  --config config/extraction/peft_dinov2_scpp_gl3d_pretrain.yaml \
  --input-root data/GL3D/train --data-name gl3d_train \
  --device cuda:0 --record-output results/extraction/gl3d_raw.json
```

Pass the resulting record to retrieval or whitening tools.
Named outputs are stored as separate streams; `--stream-output NAME=H5`
exports a selected stream. See [feature storage](../gmlab/artifacts/README.md).

Extraction defaults to FP32. `--precision` selects AMP, `--matmul-tf32` and
`--cudnn-tf32` select backend policies, and `--storage-dtype` selects the saved
dtype. Saving float32 does not make an AMP forward equivalent to FP32.

## Retrieval

[Retrieval and reranking](../gmlab/RETRIEVAL.md) covers live features, cached
features and search settings. Pair evaluation with `--gt` measures agreement
with supplied GT; geometric verification measures a different quantity.

Classical retrieval requires COLMAP and `psutil`; vocabulary fitting requires
FAISS. See [installation](../gmlab/INSTALL.md) for dependencies and model setup.
