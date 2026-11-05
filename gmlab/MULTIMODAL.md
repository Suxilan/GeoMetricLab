# Metric pipelines, modality branches and joint training

The data source supplies modality and identity. The experiment selects image
pipelines and model components. The model chooses its configured branch from the
explicit modality, never from a Dataset name, path or query/gallery role.

## Choose the smallest configuration for the task

All cases use `python -m gmlab.train --config <recipe.yaml>`. `GeoPipeline` binds
data, supervision and model-output contracts. `MultimodalPipeline` specializes
that binding for explicit modality groups. `MetricModule` executes these routes
through native Lightning; it does not decide which modality or GT a sample means.

Start with the [recipe guide](../config/gmlab/README.md) and inherit the base
that owns your experiment's contract:

| Base recipe | Pipeline | Accepted training contract |
| --- | --- | --- |
| `metric/base.yaml` | `GeoPipeline` | Tensor inputs with labels |
| `relation/base.yaml` | `GeoPipeline`, relations | Padded relation batches with valid-node masks |
| `multimodal/base.yaml` | `MultimodalPipeline` | Explicit modalities and grouped inputs |
| `joint/base.yaml` | `MultiDatasetPipeline` | A task route plus at least two named sources and mixing |

Bases expose required component slots, task-specific options and practical runtime
defaults. They inherit shared evaluation/logging policy from `runtime.yaml`.
Experiments supply their model, loss, data, transforms and actual runtime. Null
required fields deliberately fail validation; a base alone is not a runnable run.
For grouped image inputs with relation supervision, the task-specific fragment is:

```yaml
extends: ../multimodal/base.yaml
pipeline:
  modalities: [rgb, thermal]
  supervision: relations
data:
  train:
    loader:
      collate_fn:
        type: PadSubgraphs
```

Fill the remaining required slots with compatible components. The dataset must
supply modality metadata for each valid node; this does not convert heterogeneous
point/image tensors into padded image batches.

For multiple heads and independent GT, select named supervision and one retrieval
head directly; no custom output-selector function is needed:

```yaml
pipeline:
  type: MultimodalPipeline
  modalities: [rgb, thermal]
  supervision: named
  training_output: named
  retrieval_output: retrieval
```

The same output selection is used by validation, standalone evaluation and cache
extraction. Extraction recipes accept this same `pipeline` block. A named
`streams` export uses `pipeline.select_streams` instead of collapsing outputs.
The cache identity records the pipeline class and inference settings.

| Requirement | Model configuration | Data configuration |
| --- | --- | --- |
| One descriptor model | `GeoEncoder`: backbone, aggregator, BN, optional projection | `data.train.dataset` and `loader` |
| Shared encoder, different modality heads | `ModalityEncoder`: one `encoder`, named `heads` | Grouped modality collate |
| Shared/independent backbone, agg, BN, postprocessor | `NamedBranches` component pools plus `modality_routes` | Grouped modality collate |
| Different networks/input shapes, e.g. images and points | Per-modality `encoders` in `ModalityEncoder` or routed `NamedBranches` | Downstream Dataset/collate supplies each modality Tensor |
| Joint training on independent datasets | Compose the task route under `MultiDatasetPipeline.task`; keep the shared model | `data.train.sources`, each with Dataset, loader, loss and optional pipeline overrides |

Start with these complete recipes:

- `config/gmlab/relation/gl3d_supscene_peft_dinov2_scpp.yaml`: single-source model.
- `config/gmlab/multimodal/u1652_multimodal.yaml`: shared encoder, separate modality heads.
- `config/gmlab/joint/multisource_multimodal.yaml`: two independent datasets, shared
  ResNet, Avg/GeM modality aggregators and independent BN. Both example datasets
  use University1652's folder layout; change their paths before training. This is
  a composition tutorial, not an accuracy baseline or a new dataset reader.
- `config/gmlab/joint/gl3d_u1652_joint.yaml`: actual GL3D/U1652 multi-source experiment
  using a common single descriptor model and source-specific objectives.

```bash
# Validate schema and inspect the resolved recipe without loading data/weights.
python -m gmlab.train --config config/gmlab/joint/multisource_multimodal.yaml --print-config
# After setting the two dataset paths, run a finite check with the same model.
python -m gmlab.train --config config/gmlab/joint/multisource_multimodal.yaml --dev 2
python -m gmlab.train --config config/gmlab/joint/multisource_multimodal.yaml
```

In `NamedBranches`, pools instantiate modules once. Both branches in the tutorial
reference `encoder: shared`; `head: mean` versus `head: gem` selects aggregation,
and separate normalization references select independent BN. To share BN,
reference the same normalization key in both branches. To use a different
backbone, add an encoder and change that branch's encoder reference. These keys
are experiment-owned names, never dataset dispatch rules in the library.
All routed descriptor outputs must have the same final width for retrieval.

For LiDAR, declare an installed downstream component such as
`{type: my_package.models.PointEncoder, out_dim: 512}` in the encoder pool and
reference it from the LiDAR branch. Its head can be `torch.nn.Identity` if it
already returns `[N,512]`. Supply `inputs: {lidar: points}` and matching positions.
This is support for injected point encoders, not a bundled LiDAR architecture or
reader. Image transforms are not automatically applied to point data.

Recipe inheritance recursively merges mappings; changing from a single source
to `sources` must not leave inherited `dataset`/`loader` sibling fields. Start
from the corresponding task base and declare the chosen data layout.

### Routing validation and extension

Pipeline configuration cannot be mixed with `adapters` (or extraction's
`adapter`/`input_adapter`). Existing recipes without a pipeline retain the explicit
adapter API; new task bases use pipelines. Source overrides cannot change the
pipeline type or the shared model's input contract.

Configuration validation catches invalid pipeline options and missing named
retrieval selection. Assembly checks a model's declared input/output layout,
collator format and declared dataset modalities before `Trainer.fit`. Errors
identify the training source. Tensor shapes, row partitions and actual output
names are checked when first used. Undeclared custom-module contracts cannot be
proved without executing them; `--dev` exercises that boundary.

Downstream extensions subclass `GeoPipeline` or `MultimodalPipeline`, overriding
the relevant `prepare_training`, `prepare_inference`, or selection method. For
example, an explicit cross-modality-only candidate policy can reuse the existing
adapter without changing the model or training loop:

```python
# In an installed downstream module, e.g. my_package.pipeline.
from gmlab.pipeline import MultimodalPipeline
from gmlab.training.adapters import cross_modal_id_batch


class CrossViewPipeline(MultimodalPipeline):
    def prepare_training(self, batch, *, layout=None):
        return cross_modal_id_batch(batch)
```

Use `pipeline.type: my_package.pipeline.CrossViewPipeline`; select the compatible
rank-local objective (`distributed: false`, no additional miner) required by that
candidate policy. Subclasses adding inference settings must extend
`inference_identity()` so cache identities retain their meaning. No new registry
entry or core training edit is required for an installed import path.

New training checkpoints record resolved pipeline configuration and reject changed
routing on resume. Older checkpoints without this field retain the existing model,
source and vocabulary checks; their historical routing cannot be reconstructed
from a field they never stored. Model-only initialization remains separate.

## Modality branches and multiple heads

A modality selects one network branch. Branches may use CNNs, ViTs, point
encoders or other injected `nn.Module` implementations. Multiple heads belong
inside a branch; they do not route one modality through other network branches.

`ModalityEncoder` accepts `inputs: {modality: Tensor[Nm,...]}` and
`positions: {modality: int64[Nm]}`. The existing `images` key remains supported.
Positions partition `[0,N)` exactly. Each modality head returns one `[Nm,D]`
Tensor or named `[Nm,Dk]` Tensors. Present modalities must expose the same output
names and width per output. Different outputs may have different widths.
Results retain original row order; absent modalities execute no network or head.

```python
from torch import nn
import timm
from gmlab.models.multimodal import ModalityEncoder


class PointEncoder(nn.Module):
    """Example downstream encoder: [N,P,3] points -> [N,128]."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(3, 128), nn.ReLU())

    def forward(self, points):
        return self.features(points).mean(dim=1)


class DescriptorHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.retrieval = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64))
        self.detail = nn.Sequential(nn.Linear(128, 32), nn.BatchNorm1d(32))

    def forward(self, features):
        return {"retrieval": self.retrieval(features), "detail": self.detail(features)}


# Random initialization for a self-contained composition example.
cnn_encoder = timm.create_model("resnet18", pretrained=False, num_classes=128)
vit_encoder = timm.create_model(
    "vit_tiny_patch16_224", pretrained=False, num_classes=128
)
# These names are application configuration, not library roles.
shared_heads = DescriptorHeads()
model = ModalityEncoder(
    encoders={"rgb": cnn_encoder, "thermal": vit_encoder, "lidar": PointEncoder()},
    heads={"rgb": shared_heads, "thermal": shared_heads, "lidar": DescriptorHeads()},
    normalize=True,
)
```

RGB and thermal share heads and BN state; LiDAR has independent heads and BN.
Shared modules execute separately per present group, so BN updates follow group
order. To share selected layers only, reference those modules inside downstream
heads. YAML supports the same sharing through local `modules` and `{ref: name}`.
For a single output per modality, `NamedBranches` with `modality_routes` exposes
encoder/head/BN/postprocessor pools directly. Its unrouted mode can also compose
several heads **inside** one modality branch; it does not infer modality routing.

## Compose with multiple training datasets

Inherit `joint/base.yaml` and select the task under `pipeline.task`. For named
head outputs, set `pipeline.task.supervision: named` and
`pipeline.task.training_output: named`. The batch contains `inputs`, `positions` and `supervision`.
Each supervision entry addresses the original rows and retains its own GT.
`BoundObjective(output="retrieval", arguments={"labels": "identity"},
objective=...)` binds a named output and GT to a PML objective; `GeoMetricLoss`
combines metric and auxiliary objectives. Tensor outputs use `training_output: descriptor`.

Each `data.train.sources` entry independently declares its Dataset, loader,
loss and optional pipeline overrides. For example, a source can set
`pipeline: {supervision: relations}` while another uses labels. `MultiSourceLoader`
interleaves source batches; the model
routes modalities within a batch. A source can contain only a subset of the
model's modalities. Configure native Lightning unused-parameter handling when
branches may be absent in distributed training.

`MetricModule` is the Lightning runtime. Python callers can compose
`TrainingSource` records and pass the module to `Trainer.fit`, or subclass it in
their downstream package for custom lifecycle hooks. The standard YAML entrypoint
constructs this same module with pipeline-bound routes. Dataset decoding belongs
to the Dataset; the pipeline prepares the model contract and the model executes
its selected branches. New modality or dataset names require no
special cases in the training loop. Objectives remain within metric learning;
there is no additional registry for arbitrary unrelated task types.

Each step consumes **one source batch**. Source budgets are the native loader
lengths, optionally capped by `pipeline.mixing.steps_per_source`, and deterministically
interleaved; batches from different datasets are not
concatenated and do not automatically form cross-dataset positive/negative pairs.
Labels are local to each source's objective. Declare distinct `source_scope`
values when separate datasets use the same local physical IDs. Sampling exposure
and objective weights have different meanings: configure dataset/sampler lengths
or explicit batch caps for exposure, and `GeoMetricLoss` weights for loss contributions.
The [joint tutorial](../config/gmlab/joint/README.md) specifies scheduling and resume behavior.

### Batch contracts implemented by the pipelines

| Input/target layout | Pipeline selection | Inference |
| --- | --- | --- |
| Existing grouped image collates with labels | `MultimodalPipeline`, labels | Grouped inputs and stable indices |
| Grouped arbitrary Tensors with named GT | `MultimodalPipeline`, named | Grouped inputs and stable indices |
| SupScene padded relation batches | Either pipeline, relations | Tensor or grouped inputs according to pipeline |

For heterogeneous inputs, a downstream collate returns:

```python
{
    "inputs": {"rgb": rgb_images, "lidar": point_clouds},
    "positions": {"rgb": rgb_rows, "lidar": lidar_rows},
    "supervision": {"identity": aligned_labels},  # training only
    "index": manifest_indices,  # inference only
}
```

Positions partition the original batch rows. Each Tensor has a batch axis; the
encoder owns remaining dimensions. Named GT remains aligned to original rows.
Pipeline inference requires no GT and preserves the dataset's stable indices.
Variable-size point clouds need downstream padding/masking or an explicit adapter;
this grouped Tensor contract does not silently support arbitrary ragged objects.
For a SupScene recipe with a grouped model, select `MultimodalPipeline` with
`supervision: relations`. It reuses valid-node decoding and groups nodes by the
batch's explicit modality metadata, preserving the same supervision and row order.
Missing modality metadata is an error; the pipeline does not invent it.

Use the same `pipeline.retrieval_output` selection and
inference precision for live and cached retrieval; see [RETRIEVAL.md](RETRIEVAL.md).

The installed library has no fixed satellite/ground dual-tower model. Historical
checkpoint layout conversion must be explicit in the downstream application.

## Run the component recipe

`config/gmlab/multimodal/u1652_multimodal.yaml` uses a shared GeoEncoder and separate satellite
and drone heads. It is a runnable component recipe, not a reproduced accuracy
baseline. Set paths, transforms, learning rates and hardware for your experiment.
Its existing held-out collection is used for validation/selection; evaluating the
selected checkpoint there is not an independent test-set accuracy claim.

```bash
python -m gmlab.train --config config/gmlab/multimodal/u1652_multimodal.yaml
python -m gmlab.evaluate --config config/gmlab/multimodal/u1652_multimodal.yaml \
  --split valid --run-dir results/gmlab/u1652_multimodal --weights best \
  --output results/gmlab/u1652_multimodal/retrieval.json
```

## Select transforms per split and modality

Each split can reference one named transform sequence. For distinct modality
transforms, use the following explicit mapping:

```yaml
transform:
  by_modality:
    satellite: satellite_train
    drone: drone_train
```

Names refer to ordered torchvision operation lists in `data.transforms`. Train
and eval select their own transforms. `build_image_pipelines` returns either one
callable or a modality-to-callable mapping; the reader applies the selected
callable after the reader decodes each image through `GeoDataset`. Missing modalities fail rather than silently
using a shared default. Custom operations can use registered or qualified types.

Transforms run independently per image. Paired geometric supervision must supply
an explicit synchronized transform; grouping images does not establish geometric
correspondence or justify applying the same random crop.

## Dataset and batch contract

`gmlab.datasets.utils.geo_dataset.GeoDataset.prepare_rows` is the shared selected-row
lifecycle: validate physical identity and declared modality, load each selected
input, apply its modality transform, and assign its position in the selection.
The public `transform` configuration is the single execution/provenance source;
between-run pipeline replacements do not leave a second stale transform map.
It does not require an integer request, a modality-candidate enumeration or a
particular sampling algorithm. It consumes iterables lazily to preserve RNG order.
`row_position` is reserved and must match selection order if supplied.

```text
GeoDataset
├── University1652InstanceDataset
├── SubgraphDataset
│   ├── GL3DSubgraphDataset
│   └── SceneSubgraphDataset
└── ImageList
    └── RetrievalImages
```

The shared classes and checkout example readers call `prepare_rows`, directly or indirectly.
Their request and output layouts retain their own responsibilities:

| Reader | Selection and remaining responsibility |
| --- | --- |
| U1652 | Class-index candidates and optional quota policy; source discovery and RGB loading. |
| GL3D / SceneSubgraphDataset | Epoch/explicit-scene requests, graph-node sampling and group GT; only sampled rows enter preparation. |
| ImageList | Stable manifest index, path and optional crop; no GT required. |
| RetrievalImages | ImageList preparation plus separate query/gallery/GT metadata. |

The only other project-defined Dataset is training.extraction._ExtractionJob, a
native IterableDataset yielding one launcher control token per rank. It contains
no image/sample rows; the actual input loader still uses the readers above.
Native Subset views likewise delegate reads to their underlying Dataset. These
are scheduling/view mechanisms, not alternate sample-preparation pipelines.

The optional `GeoDataset.__getitem__` convenience implements integer-indexed
candidate groups; readers using it supply `__len__` and `candidates(index)`.
`candidate_sampler` belongs only to this convenience workflow. The graph sampler
stays owned by SubgraphDataset with its native `sample(graph, n_sub)` protocol.
Generic Tensor readers can retain identity `load_input`; RGB readers reuse
`load_rgb_image`. In-place transforms require fresh loaded inputs; opaque input
Tensors are not automatically cloned. External ordinary Datasets may still emit
the same row groups without inheriting the class.

```python
item = {
    "index": dataset_item_index,  # May address a whole class/group, not an image.
    "rows": [
        {
            "inputs": input_tensor,
            "sample_id": (source_scope, physical_id),
            "modality": modality_name,
            "label": class_id,  # Optional; shared class namespace across modalities.
        }
    ],
}
```

Physical IDs are explicit strings or integers in a nonempty source scope. Numeric
values do not imply unknown GT. Dataset item indices, physical identity, class
labels and row occurrences are separate. Repeated draws retain the same physical
identity and distinct rows; they are never silently deduplicated.

`datasets.utils.samples.stack_samples` and `group_samples` consume these groups:

| Batch field | Meaning |
| --- | --- |
| `images` | Stacked `[N,...]`, or modality → `[Nm,...]`; each group has one shape. |
| `labels` | Optional integer `[N]`, in original row order; present on all rows or none. |
| `sample_ids` | Physical `(scope, ID)` pairs, including repeated draws. |
| `item_indices` | Dataset item indices `[N]`, repeated for grouped samples. |
| `item_positions` | Batch item occurrence positions `[N]`; keeps two draws of the same index as separate groups. |
| `modalities` | Explicit names in original row order. |
| `modality_ids`, `modality_names` | Batch-local numeric codes and their vocabulary; not fixed sensor IDs. |
| `positions` | Modality → int64 row positions; partitions `[0,N)` and restores group outputs. |
| `row_metadata` | Extra annotations retained without inferring pairing or sequence semantics. |

Collators declare their output layout, so training selects the corresponding
processor automatically. Stacked and grouped batches retain identity/layout metadata.
The existing `multiview_id` processor remains for external tuple inputs, independently
of this mapping contract. There is no U1652-specific collate or compatibility
forwarder. Positions do not establish physical identity, positive pairs or
cross-modality synchronization.

## Declare exposure and seed constraints

The U1652 reader only discovers class/view files and loads RGB images. Its old
`n_samples` and `input_transform` options are removed: use the common `transform`
and an explicitly injected sampling policy. The migrated recipes declare the
original exposure distribution, for example:

```yaml
dataset:
  type: datasets.train.u1652.University1652InstanceDataset
  dataset_path: data/university1652/train
  views: [satellite, drone]
  sampler:
    type: ModalityQuotaSampler
    quotas: {satellite: 1, drone: 2}
    replacement: true
    missing: skip
```

The generic policy contains no U1652/sensor-name branches. `sampler: null` means
all candidates, not an implicit one-plus-N training policy. `source_scope` can be
set explicitly when a source's identity namespace must survive relocation.

Optional `seed_modality` selects one row counted within its quota;
`required_modalities` require nonempty populations and positive quotas. Required
or seed populations cannot be skipped. For other missing populations, `missing`
is `skip` or `error`. Without replacement, `shortage` is `error` or
`take_available`; replacement concerns candidate occurrences, not a guarantee of
unique physical images.

`candidate_filter(candidates, seed)` may restrict candidates after the seed draw.
Use a callable, an import path or an existing component-factory mapping for a
parameterized filter. Return original candidate views without introducing or
duplicating entries, and retain the seed. Row/group views are read-only; inputs
and nested annotation values are borrowed read-only, not deep-copied or sandboxed.
Custom samplers may select original core rows, including copies and repeated
exposures; they cannot replace identities or annotations and may only add a
boolean `is_seed`. The seed is a sampling anchor, not a loss-anchor selection or
weight. Any corresponding metric policy must be declared separately.

Default sampling consumes a modality's `random.choices` and transforms before
sampling the next modality. Candidate order follows the source, not quota-map
key order, preserving the old RNG interleave even for Python-random transforms.
Independent tests use different modality names and a seed-conditioned candidate
filter to verify this is shared behavior, not a dataset-name special case.

## Inference and sequence boundaries

GT-free `ImageList(paths, transform, modalities=...)` now uses shared preparation
while retaining its image/index output layout. Physical `sample_id` and explicit
modality accompany each row. Default-collated identity columns are restored by
`image_batch`; grouped `modality_collate` preserves row-major identities for
`modality_image_batch`. Stable manifest indices remain separate from physical
identity, including repeated paths and different crops. No GT/loss is required.
Its public path/region/per-path modality configuration remains the input-cache
identity source; the Dataset's declared modality vocabulary is a separate field.

SubgraphDataset samples nodes before calling shared preparation; it does not
expand every scene node as a candidate or build full-scene dense GT. It preserves
scene/node IDs, group overlap and padding. Returned identity/modality/row-position
metadata is checked by collate and flattened only over valid nodes by the batch
adapter. `transform=None` retains metadata-only graph operation through
`prepare_rows(..., load_inputs=False)`, with no image decoding.
New sample metadata can carry sequence/time/validity annotations, but does not
supply temporal padding, alignment or cross-window state. The existing stateless
window example is not a stateful training contract. Reset/detach/recovery and
context-dependent cache identities require separate designs; no automatic
per-image cache reuse for a new stateful model is established here.

## Model composition and objectives

`from gmlab.models import ModalityEncoder` accepts constructed modules:

```python
model = ModalityEncoder(
    encoder=shared_encoder,
    heads={"satellite": satellite_head, "drone": drone_head},
    feature_adapter=select_encoder_features,
)
```

Alternatively supply `encoders={...}` instead of `encoder`. The YAML factory
constructs these modules from component specifications. With GeoEncoder,
`feature_adapter: descriptor_features` selects its descriptor tuple element.
All heads return [Nm,D] with the same D. The model restores original row order
and optionally L2-normalizes the resulting [N,D] tensor. Different-dimensional
outputs for heterogeneous objectives can use `NamedBranches` with explicit
output/GT bindings. Without `modality_routes`, NamedBranches runs multiple heads on each input.
With explicit `modality_routes`, it selects one configured branch per input group
and restores original row order. ModalityEncoder likewise routes groups to heads.
Routed NamedBranches currently rejects affine mounting: per-route whitening
provenance and reconstruction are not yet supported.

`feature_adapters` overrides the common feature selector by modality;
`output_adapters` selects each head's descriptor. YAML can declare a local
`modules` mapping and share components via `{ref: name}`. See the named branch
and affine contracts in [API.md](API.md) for explicit BN sharing,
postprocessing and reconstruction. No data-source name selects this behavior.

A shared encoder is registered once; ordinary optimizer prefix groups can assign
different learning rates to encoder and heads. Sharing also shares buffers: each
group runs through that encoder, including its BN. This does not provide separate
per-modality BN. Use independent encoders or explicit head modules when separate
state is required. Prediction may contain a single modality.

`group_samples` supplies class IDs for all configured modalities.
`group_cross_modal_samples` additionally constructs explicit pairs between distinct
modalities: equal IDs form positives and different IDs form negatives. The current
cross-view recipe selects this collator and keeps metric candidates rank-local.
This policy is independent of which model branches share parameters.

Grouped inference uses `modality_collate`; its declared layout selects grouped
image processing automatically. Every original input index retains its row position
in the descriptor output, including batches containing different modalities.

For DDP, ensure batches activate the required branches, or explicitly use a native
unused-parameter strategy when branches may be absent. Do not add fake images or
dummy losses. Two-rank CPU tests check gradient averaging with different heads
active on each rank and `find_unused_parameters=True`. They do not prove global
cross-view candidate mining. The local cross-view objective still needs meaningful
positive/negative candidates in each rank's batch. GPU/AMP and convergence of this
new recipe are not claimed by the CPU integration tests.

## Extend without modifying the training loop

A new source implements its row/GT interpretation and selects pipelines in YAML.
A new model can consume the same grouped input as an ordinary nn.Module. No
Trainer, GeoMetricLoss or retrieval-metric branch should be added for its name.
The grouped-input contracts above allow unequal modality counts and shapes,
alternate modality names, reverse retrieval and GT-free prediction. Keep row
identity alignment explicit when implementing another source or model.
