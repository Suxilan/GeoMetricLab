# Feature artifacts

Feature extraction writes an H5 payload, an ordered sample list and a manifest.
The returned JSON record is the input to retrieval, whitening and other consumers.

## Extract and read

This CPU example needs the `artifacts` and `runtime` extras:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from gmlab.artifacts import ArtifactStore
from gmlab.datasets.utils.feature_io import open_global_features
from gmlab.datasets.utils.image_list import ImageList
from gmlab.extraction import extract_global_features

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    Image.new("RGB", (8, 8), (40, 80, 120)).save(root / "image.png")
    dataset = ImageList(
        [root / "image.png"], transforms.Compose([transforms.ToTensor()])
    )
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    record = extract_global_features(
        model=model,
        model_config={"algorithm": "rgb_mean_v1"},
        loader=DataLoader(dataset, batch_size=1),
        input_root=root,
        data_name="example_v1",
        method_name="rgb_mean",
        store=ArtifactStore(root / "cache"),
    )
    with open_global_features(record=record) as (metadata, values):
        features = values[:]  # Copy before the H5 handle closes.
        assert features.shape == (1, 3)
```

`read_global_features(record=record)` loads the features into a NumPy array.
The open variant above supports slices without loading the whole matrix.
`extract_features` also supports named global `[D]` and spatial `[C,H,W]`
streams, returning one record per stream. See the [API](../API.md#feature-extraction-and-precision)
and [extraction tool](../../tools/README.md#extract-features).

## Location and reuse

Cache location, in order of precedence: an explicit path, `GMLAB_CACHE_ROOT`,
then `cache/` in the working directory.

```text
cache/
  global_features/<data-version>/<method>/<key>/
  spatial_features/<data-version>/<method>/<key>/
  local_features/<data-version>/<method>/<key>/
  vocabularies/<data-version>/<method>/<key>/
  whitening/<data-version>/<method>/<key>/
  indexes/<data-version>/<method>/<key>/
```

Published artifacts are immutable. Changed inputs, subsets or row orders produce
a new artifact; extraction may reuse matching rows from compatible artifacts.
Each dataset keeps its own cache. Whitening can select compatible caches without
joining or modifying their files.

Reuse checks actual model state, preprocessing, selected output, input identities
and compute/storage policies. GPU, batch size and worker count are recorded but
do not independently prevent reuse. This assumes evaluation outputs do not depend
on other samples in the batch; it does not promise bit-identical recomputation.

Offline extraction defaults to FP32. AMP and TF32 are explicit compute choices
and affect compatibility. H5 storage dtype is separate: an AMP result saved as
float32 is still an AMP result. Compare online and cached evaluation using the
same compute policy.

## Records and integrity

Records identify `features`, `samples`, `manifest`, `shape`, `dtype`,
artifact keys and payload checksums. Readers verify them and preserve row order.
Pass `cache_root` when reading records after relocating a cache.
Custom readers must declare stable `feature_samples` and
`feature_preprocessing`; use a sequential loader without dropped tails.

```bash
python -m gmlab.artifacts inventory --root cache
python -m gmlab.artifacts verify --root cache --object PATH_TO_ARTIFACT
```

Writers use local POSIX locks and private staging before atomic publication.
Handled failures can retain verified progress; interrupted H5 files are not
arbitrarily repaired. NFS and multi-node publication are not supported.
There is no automatic deletion. Downloaded weights and vocabularies remain
external assets.

For explicit file exports, pass `output="features.h5"` instead of a store;
replacement requires `overwrite=True`. Named streams publish independently.
