# Workflow examples

These scripts combine the [tools](../tools/README.md) for particular experiments.
Edit the paths and recipes for your data.

Activate the environment first. Shell scripts run from the repository root;
`PYTHON_BIN` selects Python and `DRY_RUN=1` prints commands without running them.
Use `--help` for arguments.

## Train and evaluate

Choose `metric`, `relation`, `multimodal`, `joint` or `joint_multimodal`.
Prepare the [dataset](../datasets/README.md) and edit the corresponding
[recipe](../config/gmlab/README.md).

```bash
DRY_RUN=1 bash scripts/train/run_training.sh joint --dev 2
bash scripts/train/run_training.sh multimodal
bash scripts/eval/evaluate_joint.sh results/gmlab/gl3d_u1652_joint
```

Training writes to `runtime.run_dir`. The evaluation script evaluates best and
last checkpoints. `TRAIN_CONFIG` selects another compatible joint recipe.

## Fit whitening on two datasets

The default extraction recipe expects
`weights/peft_dinov2_scpp_gl3d_pretrain.pth`.
Set `EXTRACTION_CONFIG`, `GL3D_ROOT` and `U1652_ROOT` to use other inputs.

```bash
FEATURE_DEVICE=cuda:0 bash scripts/features/run_whitening_feature_extraction.sh
bash scripts/weights/fit_whitening.sh preview
bash scripts/weights/fit_whitening.sh fit \
  --record-output results/whitening/gl3d_u1652.json
```

Extraction writes `results/extraction/{gl3d,u1652}_raw.json`.
[config/whitening/example.yaml](../config/whitening/example.yaml) selects these
records; `FEATURE_LIST` selects another YAML. Record paths inside it are
YAML-relative. The model, preprocessing and output space must match across caches.

## Retrieve images

```bash
bash scripts/retrieval/rank_cached_features.sh \
  results/query.json results/gallery.json results/rankings.jsonl
bash scripts/retrieval/extract_and_build_pairs.sh \
  data/my_scene/images config/extraction/peft_dinov2_scpp_gl3d_pretrain.yaml \
  results/my_scene
```

The first script writes ranked JSONL. Its `--config` accepts
`search` and `reranking` sections as shown in the
[retrieval guide](../gmlab/RETRIEVAL.md).

The second writes `features.json` and `pairs.txt`.
`TOPK` sets the neighbor count; `FEATURE_STREAM` selects a named global stream.
`RERANKING_CONFIG` points to an algorithm mapping, such as
`{method: aqe, k: 10}`. Extra CLI arguments go to extraction.
Use the pairs with your own matching/reconstruction workflow.

## Prepare UAV retrieval metadata

```bash
python scripts/data/create_sues200_metadata.py --root data/sues200
python scripts/data/create_denseuav_metadata.py --root data/denseuav
```

These write protocol arrays and checksummed manifests beside the images.
`--metadata-root results/metadata/NAME` writes to a separate directory.
Use that directory as `metadata_path` and the original data directory as
`image_root` in the [retrieval readers](../datasets/README.md#retrieval-metadata).
SUES-200 uses the split in `data/sues200_official_split.json` beside the builder;
DenseUAV records the released forward and reconstructed reverse protocols.
