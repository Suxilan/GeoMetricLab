# Multimodal training

[base.yaml](base.yaml) uses `MultimodalPipeline`.
Set `pipeline.modalities` and configure the model branches, loss, optimizer and dataset.
Each sample's modality selects its model branch.

## Example: shared encoder, separate heads

[u1652_multimodal.yaml](u1652_multimodal.yaml) shares a ResNet encoder between
satellite and drone images, with a separate head and transform for each.
Prepare [University1652](../../../datasets/README.md#university1652), then create
`local.yaml` beside the recipe:

```yaml
extends: u1652_multimodal.yaml
data:
  train:
    dataset:
      dataset_path: /data/university1652/train
  valid:
    sources:
      drone2satellite:
        dataset:
          metadata_path: /data/university1652/test
      satellite2drone:
        dataset:
          metadata_path: /data/university1652/test
runtime:
  run_dir: results/u1652_multimodal_local
```

```bash
python -m gmlab.train --config config/gmlab/multimodal/local.yaml --print-config
python -m gmlab.train --config config/gmlab/multimodal/local.yaml
```

The example uses the test split during training. Use separate validation
manifests and set their `split` field for independent model selection.

## Changing the branches

| Setting | Where to change it |
| --- | --- |
| Modality names | `pipeline.modalities`, dataset and model routes |
| Shared or separate encoders/heads | `model` |
| Different aggregators or BN | `NamedBranches` module pools and branch references |
| Per-modality transforms | `data.transforms`, source `transform.by_modality` |
| Sampling quotas | Dataset `sampler` |
| Batch grouping | Source `loader.collate_fn` |

`ModalityEncoder` supports a shared encoder or separate encoders with modality
heads. `NamedBranches` lets branches reference encoder, head, normalization and
postprocessing pools. Reusing a pool entry shares that module and its state.

Grouped inputs have `inputs: {modality: Tensor[Nm,...]}` and
`positions: {modality: int64[Nm]}`. Positions restore the original sample order.
Modalities may have different input shapes; descriptors compared by the same
retrieval task must have the same width. A LiDAR modality can supply its own
reader, transform and point encoder through these interfaces.

See [model configuration](../../../gmlab/MULTIMODAL.md) for full branch examples.
To combine this with several datasets, use
[joint/multisource_multimodal.yaml](../joint/multisource_multimodal.yaml).
