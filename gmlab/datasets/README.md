# Shared data utilities

This package provides dataset bases, sampling, transforms, collation and static
GT helpers. Concrete readers live in the repository's [datasets/](../../datasets/README.md).

| Component | Use |
| --- | --- |
| `utils.geo_dataset.GeoDataset` | Select candidates, load inputs, apply modality transforms and preserve identities |
| `utils.sampling.GeoEpochSampler` | Epoch requests and distributed sampling |
| `utils.modality_sampling.ModalityQuotaSampler` | Per-modality quotas and missing-candidate policy |
| `utils.samples` | Flat or modality-grouped batches from sample rows |
| `utils.supscene` | Scene graphs, subgraph sampling, relation GT and padding |
| `utils.image_list.ImageList` | Indexed image inference without GT |
| `utils.retrieval_metadata.RetrievalImages` | Image manifests with query/gallery roles and retrieval GT |
| `utils.feature_io` | Verified feature-cache readers |

## Sample rows

A `GeoDataset` reader implements `__len__` and `candidates(index)`.
Candidates are grouped by modality. Each row contains:

- `inputs`: a raw input or locator, decoded by `load_input(row)`.
- `sample_id`: `(source_scope, physical_id)`, stable across repeated sampling.
- `modality`: one of the declared modality names.
- `label`: an integer class ID when using label supervision.

The base returns `{"index": index, "rows": [...]}` after sampling and transforms.
`stack_samples` produces one tensor batch; `group_samples` groups tensors by
modality and retains their original positions. Class IDs, physical image IDs and
dataset indices have different meanings.

For scene relations, use `SubgraphDataset` or `SceneSubgraphDataset` and
`PadSubgraphs`. Keep valid-node masks and known/unknown pair masks explicit;
a non-transitive relation graph cannot be replaced by class labels.

Retrieval items supply model inputs and stable indices. Dataset-level
`eval_meta` supplies query/gallery indices and GT when available.
See [the examples](../../datasets/README.md) for file layouts and configuration.
