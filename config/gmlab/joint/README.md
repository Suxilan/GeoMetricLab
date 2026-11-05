# Joint training

[base.yaml](base.yaml) uses `MultiDatasetPipeline`.
It needs a task pipeline, a mixing policy, a model, an optimizer and at least two
named `data.train.sources`. Each source supplies a dataset and loader.
Use a global loss or give each source its own loss.

| Example | Task |
| --- | --- |
| [gl3d_u1652_joint.yaml](gl3d_u1652_joint.yaml) | Shared model, GL3D/U1652 relation supervision, separate losses |
| [multisource_multimodal.yaml](multisource_multimodal.yaml) | Two collections, shared encoder, separate modality aggregation and BN |

## Example: joint multimodal training

The multimodal recipe uses two collections in the
[University1652 folder layout](../../../datasets/README.md#university1652).
Create `local.yaml` beside it:

```yaml
extends: multisource_multimodal.yaml
pipeline:
  mixing:
    policy: interleave
    steps_per_source: null
data:
  train:
    sources:
      collection_a:
        dataset:
          dataset_path: /data/collection_a/train
      collection_b:
        dataset:
          dataset_path: /data/collection_b/train
  valid:
    sources:
      drone2satellite:
        dataset:
          metadata_path: /data/collection_a/test
      satellite2drone:
        dataset:
          metadata_path: /data/collection_a/test
runtime:
  run_dir: results/joint_local
```

```bash
python -m gmlab.train --config config/gmlab/joint/local.yaml --print-config
python -m gmlab.train --config config/gmlab/joint/local.yaml
```

## Task and source settings

`pipeline.task` holds a `GeoPipeline` or `MultimodalPipeline` configuration.
A source can override task options such as supervision in its `pipeline` mapping.
Its batches must match the shared model's input format. A source-specific `loss`
replaces the global loss. Distinct collections with overlapping local IDs need
distinct `source_scope` values.

One training step consumes one source batch. `interleave` distributes source
batches through the epoch. With `steps_per_source: null`, it visits every
available batch once. To limit batches per source, name every source, for example
`{collection_a: 100, collection_b: 50}`; caps must fit the available loaders.
Changing exposure and changing loss weights have different effects.

Sources share the model and optimizer. GT stays local to each source; the
scheduler does not create cross-source positive or negative pairs.
Validation/test sources run separately, and the checkpoint monitor names one
metric, such as `val/drone2satellite/mAP`.
