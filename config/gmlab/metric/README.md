# ID-supervised training

[base.yaml](base.yaml) uses `GeoPipeline` with label supervision.
Supply `model`, `loss`, `optimizer` and `data.train.dataset`.

For a dataset returning `(inputs, label)`, PyTorch's default collation produces
inputs `[N,...]` and labels `[N]`. Readers returning structured sample rows use
`stack_samples`, which returns a mapping with images, labels and identities.
The model produces descriptors `[N,D]`. Choose batches with the positive and
negative examples required by the loss.

## Example: University1652

[u1652_labels.yaml](u1652_labels.yaml) uses ResNet-18, GeM and class IDs.
Follow the [dataset setup](../../../datasets/README.md#university1652), then create
`local.yaml` in this directory:

```yaml
extends: u1652_labels.yaml
data:
  train:
    dataset:
      dataset_path: /data/university1652/train
    loader:
      batch_size: 16
  valid:
    dataset:
      metadata_path: /data/university1652/validation
  test:
    dataset:
      metadata_path: /data/university1652/test
runtime:
  run_dir: results/u1652_local
```

```bash
python -m gmlab.train --config config/gmlab/metric/local.yaml --print-config
python -m gmlab.train --config config/gmlab/metric/local.yaml
```

For training without retrieval splits, inherit `base.yaml` and declare only
`data.train`. For several datasets, use the [joint base](../joint/README.md).
Named GT and output bindings are covered in [the interface guide](../../../gmlab/MULTIMODAL.md).
