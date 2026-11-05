# Training configurations

Training recipes inherit one of the following bases and specify the model,
loss, optimizer and data.

| Task | Base and example |
| --- | --- |
| One dataset with class IDs | [metric](metric/README.md) |
| One dataset with relation GT | [relation](relation/README.md) |
| Modality-specific model branches | [multimodal](multimodal/README.md) |
| Joint training on several datasets | [joint](joint/README.md) |

The relation base uses `GeoPipeline` with relation supervision. Joint training
composes either `GeoPipeline` or `MultimodalPipeline` under `pipeline.task`.
The pipeline validates required fields and checks the declared data and model
formats.

## Inheritance

Put a new YAML beside the example you want to change:

```yaml
extends: u1652_labels.yaml
optimizer:
  lr: 0.00003
runtime:
  run_dir: results/my_experiment
  trainer:
    max_epochs: 30
```

`extends` is relative to the YAML file. Mappings merge recursively; lists,
scalars and `null` replace inherited values. Data, weight and output paths are
relative to the working directory.

Use a base to start a new task; use a complete recipe to make a variant of that
experiment. Required `null` fields in a base must be filled. Check inherited
validation paths, model initialization and hardware settings too.

## Run

Run the supplied dataset examples from the repository root:

```bash
python -m gmlab.train --config config/gmlab/metric/u1652_labels.yaml --print-config
python -m gmlab.train --config config/gmlab/metric/u1652_labels.yaml --dev 2
python -m gmlab.train --config config/gmlab/metric/u1652_labels.yaml
```

`--dev` limits batches and disables checkpointing and tracking. Partial validation
does not report retrieval metrics. To resume a full run, add
`--resume /path/to/checkpoints/last.ckpt`.
For model weights alone, use `initialization.checkpoint` in a new experiment.

Concrete readers are in [datasets/](../../datasets/README.md); shared component
interfaces are documented in the [API](../../gmlab/API.md).
