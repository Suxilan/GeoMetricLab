# Dataset examples

Readers for GL3D, University1652, SUES-200 and DenseUAV, using the shared bases
and utilities in [gmlab.datasets](../gmlab/datasets/README.md).
Training examples cover GL3D relation GT and University1652 IDs or relation GT.

```text
datasets/
  train/
    gl3d.py
    u1652.py
  valid/
    gl3d.py
    u1652.py
    uav.py
```

Run examples from the repository root. This directory is included in the source
archive, but is not installed as a global Python package. Recipes use explicit
paths such as `datasets.train.gl3d.GL3DSubgraphDataset`.
For your own installed reader, use its package path instead.
Image files and local data links remain in `data/`.

## University1652

The ID reader expects:

```text
data/university1652/train/
  satellite/<class_id>/*.jpg
  drone/<class_id>/*.jpg
```

Classes are discovered from the first configured view. The reader supplies
candidates; `ModalityQuotaSampler` controls how many images to draw from each
view. A loader batch counts classes, while the resulting tensor batch counts
selected images.

Start with [the ID recipe](../config/gmlab/metric/README.md) or
[the multimodal recipe](../config/gmlab/multimodal/README.md).
Set the training directory and prepare the retrieval manifests below.

For relation supervision, `University1652SubgraphDataset` reads
`gt_root/<class_id>/gt.npz`. GT node names are relative to the training root,
for example `drone/0001/image.jpg`. Supply an explicit node order and aligned
sparse edges as in the GL3D format below.
See [the relation recipe](../config/gmlab/relation/u1652_global_circleloss_a6k.yaml).

## GL3D

The training reader expects prepared scene annotations:

```text
data/GL3D/train/<scene>/
  images/
  image_index.json
  gt.npz
```

`image_index.json` can be a list of image names in GT node order.
`gt.npz` contains aligned one-dimensional `row`, `col` and `weight` arrays.
Indices refer to that image order. Weights are clipped to [0,1], duplicate
undirected edges use their maximum, and missing edges inside a scene have zero
overlap. These annotations must be prepared before training.

`GL3DSubgraphDataset` discovers scenes and delegates sampling to the shared
`SubgraphDataset`. Set `scene_names` for an explicit subset.
The [GL3D training guide](../config/gmlab/relation/README.md) covers the
PEFT DINOv2 + SCPP recipe.

## Retrieval metadata

The GL3D and University1652 readers consume three NumPy files:

| File | Contents |
| --- | --- |
| `<prefix>_dbImages.npy` | One-dimensional gallery image-name array |
| `<prefix>_qImages.npy` | One-dimensional query image-name array |
| `<prefix>_gt.npy` | One list of positive gallery positions per query |

Names are relative to `image_root`, which defaults to `metadata_path`.
GT uses zero-based positions in the gallery array, not image IDs.
Store variable-length GT as a NumPy object array; load only trusted annotations.

GL3D defaults to prefix `gl3d_test` and excludes a query's own gallery image.
University1652 uses `u1652_<direction>`, with directions
`drone2satellite`, `satellite2drone`, `satellite2street` or `street2satellite`.
Set `split` explicitly to describe the source split.

For example, the following is a validation dataset section:

```yaml
dataset:
  type: datasets.valid.u1652.u1652_retrieval
  metadata_path: data/university1652/validation
  image_root: data/university1652
  split: validation
  direction: drone2satellite
```

SUES-200 and DenseUAV readers in `valid/uav.py` also verify their protocol
manifests. Use the [metadata scripts](../scripts/README.md#prepare-uav-retrieval-metadata)
to generate those files.

## Adding a dataset

A plain PyTorch dataset returning `(inputs, label)` can use the metric pipeline
directly. For shared identity, sampling and modality handling, subclass
`GeoDataset`. Here is a minimal reader for a JSON list of
`{"path": "images/a.jpg", "label": 0}` entries:

```python
import json
from pathlib import Path

from gmlab.datasets.utils.geo_dataset import GeoDataset
from gmlab.datasets.utils.image_list import load_rgb_image


class ManifestDataset(GeoDataset):
    def __init__(self, root, manifest, transform=None, source_scope="images"):
        super().__init__(
            modalities=("rgb",),
            transform=transform,
            identity_namespace=source_scope,
        )
        self.root = Path(root)
        self.entries = json.loads(Path(manifest).read_text())

    def __len__(self):
        return len(self.entries)

    def candidates(self, index):
        entry = self.entries[index]
        return {
            "rgb": [
                {
                    "inputs": self.root / entry["path"],
                    "sample_id": (self.identity_namespace, entry["path"]),
                    "modality": "rgb",
                    "label": entry["label"],
                }
            ]
        }

    def load_input(self, row):
        return load_rgb_image(row)
```

Save this in `datasets/train/my_dataset.py` and configure
`type: datasets.train.my_dataset.ManifestDataset`, `root` and `manifest`.
Use `collate_fn: stack_samples` and an image transform with the
[metric base](../config/gmlab/metric/README.md).
Choose batches containing positives for your loss.
