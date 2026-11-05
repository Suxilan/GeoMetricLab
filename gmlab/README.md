# Using gmlab

Install the package from the [repository root](../README.md#installation).
The library separates data preparation, model computation and metric objectives.
Use these components in a regular PyTorch training loop, or combine them through
the supplied pipelines for training, evaluation and feature extraction.

## Model and loss

This example runs on CPU without pretrained weights:

```python
import torch
from torch import nn

from gmlab.losses import GeoMetricLoss
from gmlab.models import GeoEncoder

model = GeoEncoder(
    backbone=nn.Conv2d(3, 8, kernel_size=1),
    use_bn=False,
)
images = torch.rand(4, 3, 32, 32)
labels = torch.tensor([0, 0, 1, 1])

descriptors, auxiliary = model(images)  # [N, D], auxiliary outputs
criterion = GeoMetricLoss()
result = criterion(descriptors, labels=labels)
result["loss"].backward()
```

`GeoEncoder` combines a backbone, pooling, optional BN and whitening, and L2 normalization.
`GeoMetricLoss` combines configured metric objectives and auxiliary losses.
The example uses class labels. Explicit pairs and triplets use the same loss
interface; prepared relation matrices use `MetricObjective`, as shown below.

## Supervision

These examples use four descriptor rows, A–D, and the `runtime` extra.
Choose the representation that matches your annotations; a class ID is optional
for objectives that support explicit pairs or triplets.

```python
from gmlab.training.metric import MetricObjective

features = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [-1.0, 0.0]],
    requires_grad=True,
)
criterion = GeoMetricLoss(
    primary_objective=MetricObjective("MultiSimilarityLoss", distributed=False)
)
```

**Pairs.** PML uses four index vectors: positive anchors, their positive
references, negative anchors, and their negative references.
Here A–B and B–C are positive; A–C and A–D are negative, in both directions.

```python
pairs = (
    torch.tensor([0, 1, 1, 2]),  # Positive anchors.
    torch.tensor([1, 0, 2, 1]),  # Positive references.
    torch.tensor([0, 2, 0, 3]),  # Negative anchors.
    torch.tensor([2, 0, 3, 0]),  # Negative references.
)
pair_loss = criterion(features, indices_tuple=pairs)["loss"]
```

**Triplets.** Specify aligned `(anchor, positive, negative)` index vectors and
choose a triplet objective. This example constrains A to be closer to B than C.

```python
triplets = (torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
triplet_criterion = GeoMetricLoss(
    primary_objective=MetricObjective("TripletMarginLoss", distributed=False)
)
triplet_loss = triplet_criterion(features, indices_tuple=triplets)["loss"]
```

**Relation matrices.** The same pair annotations can be stored as masks.
This example uses `+1` for positive, `-1` for negative, and `0` for unknown or
ignored entries. This is an example encoding; the adapter below explicitly
decodes it into positive and negative masks.

```python
from gmlab.datasets.utils.relations import prepare_relations

matrix = torch.tensor([
    [ 0,  1, -1, -1],
    [ 1,  0,  1,  0],
    [-1,  1,  0,  0],
    [-1,  0,  0,  0],
])
targets = prepare_relations(pos_mask=matrix.eq(1), neg_mask=matrix.eq(-1))
matrix_loss = criterion(features, supervision={"relations": targets})["loss"]
torch.testing.assert_close(pair_loss, matrix_loss)
matrix_loss.backward()
```

The pair and matrix examples express the same constraints and use the same loss.
An exact triplet list specifies its own constraints; expanding it into all
possible combinations of pairs may change the objective. Class-dependent losses
such as CosFace still require labels. See [relation training](../config/gmlab/relation/README.md)
for dataset/collate setup and [the API](API.md) for independent GT/output bindings.

## Training and extensions

Choose a [pipeline base](../config/gmlab/README.md) and supply the model, loss,
optimizer and data. Lightning runs training, validation and checkpointing.
`GeoPipeline` connects batches, model outputs and loss inputs;
`MultimodalPipeline` handles modality inputs, and `MultiDatasetPipeline`
schedules independent sources.

Built-in components use registered names in YAML. Your own components can use
an import path:

```yaml
dataset:
  type: my_package.readers.MyDataset
  root: /path/to/data
```

Constructors receive configured arguments; data loading also injects the selected
transform. Python callers can pass component instances directly.
See the [dataset tutorial](../datasets/README.md#adding-a-dataset).

## Reference

| Area | Documentation |
| --- | --- |
| Component construction, training and model interfaces | [API](API.md) |
| Dataset bases, samples and collation | [Data utilities](datasets/README.md) |
| Modalities, branches and named outputs | [Multimodal models](MULTIMODAL.md) |
| Online/offline retrieval and reranking | [Retrieval](RETRIEVAL.md) |
| Feature storage and reuse | [Artifacts](artifacts/README.md) |
