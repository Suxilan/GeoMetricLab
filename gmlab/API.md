# Component contracts

GeoMetricLab composes PyTorch models, PyTorch Metric Learning objectives and
Lightning training. YAML declares components; Python callers can inject the same
objects directly. See [installation](INSTALL.md), [training](README.md),
[multimodal inputs](MULTIMODAL.md) and [feature storage](artifacts/README.md).

## Configuration

Recipes contain `model`, `optimizer`, `data` and `runtime`, with optional
global `loss`, `pipeline`, `scheduler`, `evaluation` and `initialization`. A global loss is
required unless every named training source declares its own.
Built-in components use registered short `type` names; external components use
explicit import paths.

## Pipeline routing

`gmlab.pipeline.GeoPipeline` binds Tensor models to label, relation or named GT
contracts. `MultimodalPipeline` handles grouped inputs, including grouping valid
relation nodes without changing GT. Both supply the actual training/inference
adapters used by native `MetricModule`, standalone evaluation and extraction.
`MultiDatasetPipeline` composes one of these task routes with named sources and
finite interleaving. `build_pipeline(config)` constructs the experiment owner;
`resolve_pipeline_route(config, source=None)` resolves the executable task route,
including source-local options. Pipeline-owned validation runs before data/model
construction; native constructors retain validation of their own arguments.
See the [recipe tutorials](../config/gmlab/README.md) and
[extension contracts](MULTIMODAL.md) for configuration.

## Model and retrieval components

Backbones include ResNet, DinoV2, PEFTDinoV2, DinoV3, PEFTDinoV3, ConvNeXt
(DINOv3), SwinTransformer and DepthAnythingV2Backbone. External implementations
and weights remain explicit assets. `pretrained=False` means random initialization.
Depth-Anything uses native attention by default; `native_attention=False` selects
the upstream xFormers path where supported. DINOv2's upstream CPU fallback can
be selected with `XFORMERS_DISABLED=1` before importing the external repository.

Aggregators include Avg, CLS, GeM, SCPP, SuperGeM, BoQ, Salad, NetVLAD,
GhostVLAD, SuperVLAD, CosPlace, EigenPlace, CricaVPR, CFP, SoftPCFP, RegVPR,
DGeM and CGeM. They return `(descriptor [N,D], auxiliary)` and expose
`out_channels`. `DinoAggregation(aggregator=...)` passes DINO's tuple to
CLS-dependent aggregators; patch-only heads use the `patch_tokens` adapter.

`gmlab.rerank.apply_reranking(method, queries, gallery, **options)` accepts
`aqe`, `dba`, `dfs`, `gnn`, `krnn`, or an injected callable. Inputs are finite
`[Q,D]` / `[G,D]` on one device; refinement preserves row order and may change
width. GNN still requires quadratic adjacency storage. No external
GPU-Re-Ranking checkout or CUDA extension is imported. The algorithms reuse
the shared exact-search backend.

`evaluate_retrieval(..., reranking={"method": "aqe", "k": 10})` applies refinement
before ranking, without changing roles or GT. Reranking is opt-in and its
candidate population and parameters belong to the experiment configuration.
`gmlab.retrieval.retrieve(queries, gallery, k=20, reranking=...)` returns actual
ranked gallery positions without requiring GT. Evaluation, result export and
pair construction share this retrieval path. See [RETRIEVAL.md](RETRIEVAL.md)
for online/cache usage and the population dependence of each implementation.

`WeightedSupConLoss` is a PML extension with finite nonnegative `[A,R]`
positive-pair weights. Weights normalize the positive log-probability mean per
anchor; they do not weight the negative denominator. PML retains its forward
lifecycle, distance, reducer, statistics and embedding regularization.
Constructor arguments must match the selected component's signature.

`extends` names one YAML file relative to the declaring file. Mappings merge
recursively; lists and scalars replace their inherited values. Other paths remain
relative to the launch directory. `--print-config` resolves and validates a recipe
without loading samples or model weights.

Each data split declares `dataset`, optional `transform` and native DataLoader
options in `loader`. Named `data.valid.sources` and `data.test.sources` give each
retrieval source its own dataset, preprocessing and protocol. Monitor
names include the source, for example `val/drone2satellite/mAP`.
Example readers live in the checkout's `datasets/train` and `datasets/valid`.
Recipes reference them by Python path, for example
`datasets.train.gl3d.GL3DSubgraphDataset`. Run these examples from the repository
root. Your own installed readers can use their package paths in the same way.
The core library contains shared [dataset utilities](datasets/README.md), not
dataset-specific registrations. Collators use the built-in names `stack_samples`,
`group_cross_modal_samples` or `modality_collate`.

`data.train.sources` declares independent training datasets, loaders, losses and
optional `adapters.batch`/`adapters.embedding` overrides. Source names are route
keys, never GT types. One optimizer updates the shared model and all registered
objectives. The source's native collator constructs its batch; the source adapter
selects model input and criterion arguments. A source-local loss overrides the
global fallback. One step consumes one source, so cross-source pairs are not
inferred. Training metrics use `train/<source>/...`; validation sources remain
independent and checkpoint monitoring selects one explicit metric. Python callers
use `TrainingSource(criterion, batch_adapter, embedding_adapter)` records from
`gmlab.training.module`. A single legacy criterion is normalized into one default
record, preserving its existing checkpoint parameter keys. Named sources keep
their own registered criterion parameters; both use the same step execution.
Per-source step curves contain only that step's active source, emitted directly
to the injected Lightning loggers; epoch curves use native Lightning aggregation.

The joint GL3D/U1652 recipe uses MultiSimilarityLoss for GL3D and CircleLoss
(`m=0.4`, `gamma=80`) for U1652; each retains its configured ranking auxiliary.
It validates four separately named retrieval protocols every five epochs and at
the final step: GL3D, University1652 drone-to-satellite, SUES-200 mixed-height
drone-to-satellite and DenseUAV all-height drone-to-satellite. Only GL3D mAP
selects checkpoints. `datasets.valid.uav.sues200_retrieval` also accepts
`height: 150/200/250/300`; it and `datasets.valid.uav.denseuav_retrieval` accept
`direction: satellite2drone` for the reverse protocol. Existing JSON annotation
checksums and GT are verified; no training dataset is constructed for independent
evaluation. These held-out protocols used during training are not an untouched
final test set.

`GeoDataset.sampling_multiplier` defaults to 1.0 and supports positive finite
fractions. A 0.5 multiplier takes a shuffled half-round; 1.5 takes one complete
round and a new shuffled half-round. Half-rounds in consecutive epochs are
independent selections, not complementary halves. `shuffle: false` preserves
the reader's deterministic base order. One `GeoEpochSampler` consumes the
Dataset's `epoch_requests`; grouped readers override a base round, not multiplier
or rank-partition logic. Its `distributed_sharding` declaration identifies
rank-aware samplers without a reader-type whitelist. Adaptive partial-scene
plans with unequal scene exposure are rejected because the budget can vary.
Each source's effective DataLoader length,
including batch size, rank partition and `drop_last`, defines its epoch budget.
The shared training loader interleaves those budgets deterministically and logs
planned batches, completed batches and source shares. Sampling exposure does not
change loss weights. Multi-rank training requires rank-aware source samplers;
source-specific parameters require DDP unused-parameter detection. Declare
`strategy: ddp_find_unused_parameters_true` explicitly for multi-source DDP;
the runtime validates the actual DDP model without rewriting strategy selection.
This conservative strategy supports conditional or trainable source objectives;
it does not imply that a shared-encoder recipe has any source-specific head.

Named source loader seeds are derived from the runtime seed and stable source
name, independently of mapping order. Explicit sampler `seed: null` inherits the
loader seed; an integer override remains authoritative. The revised local-RNG
plans are deterministic under this version, but are not promised to reproduce
the historical scene sampler's exact permutation.

For a named training recipe, `runtime.experiment_name` derives
`results/gmlab/<experiment_name>` and the SwanLab experiment name. An explicit
`runtime.run_dir` remains available for callers that need another location.

Standard recipes need no manual batch/output conversion section. Collators declare
`batch_format`; training assembly selects its processor. `PadSubgraphs` declares
padded scene graphs; `stack_samples` declares flat labelled rows; `group_samples`
declares modality-grouped rows. External Python callers can still inject processors
into `MetricModule`. Tensor outputs, GeoEncoder tuples and explicit `embeddings`
mappings share descriptor selection; multi-output objectives select named outputs.

`--dev` runs one batch per native Lightning loop; `--dev N` runs up to N.
Model, sampling, transforms, batch size, devices and precision stay as configured.
Tracking, checkpoint saving and early stopping are disabled. Partial validation
runs inference but does not report full-collection retrieval metrics. Each run
writes its record under a fresh `<run_dir>/dev/run-*` directory. `--dev --print-config`
shows the execution overrides without constructing datasets or creating outputs.

## Inputs and supervision

Inference datasets provide model inputs and stable row indices. Query/gallery
membership and retrieval GT belong to dataset metadata; pure extraction needs
neither labels nor GT. Data components own static identities, annotations,
sampling and transforms, never optimizers or online GPU queues.

Descriptor tensors have shape `[N,D]`. Class labels have shape `[N]`; native PML
`indices_tuple` values index descriptor rows. Relations retain their own knownness
and positive/negative masks. A non-transitive relation graph is not a class label.
SupScene inputs use `[B,N,C,H,W]` images, `[B,N,N]` overlap and explicit valid-node
masks. Missing edges within the existing scene GT mean negatives; absent relations
between different subgraphs are resolved from the registered full-scene GT
when available; otherwise they remain unknown. Cross-scene negatives
require reliable identities and the Dataset exclusivity declaration. SubgraphDataset
provides this declaration; its recipes do not need an exclusive_scenes switch.

`GeoMetricLoss` combines injected primary, additional metric and auxiliary
objectives with explicit weights. Independent GT is routed to each objective;
one target never supplies missing labels or relations for another. Model tensors
retain gradients for objectives; logging consumes detached values.

Objectives declare the same-name supervision fields they consume, so ordinary
single-GT recipes configure the objective directly. `BoundObjective` is available
for explicit named output selection or remapping independent GT fields; it is not
needed for identity mappings. Overlap ranking uses PML with within-subgraph
triplets and a rank-local mean. It has no backend or distributed switch. DDP still
averages parameter gradients. At an exactly zero hinge, PML's zero subgradient
differs from PyTorch margin_ranking_loss; this boundary is part of its semantics.


`MetricObjective` prepares candidates and executes the selected PML loss.
Pair weights and objective weights have different meanings. Supported pair
weights are finite nonnegative `[N,N]` tensors in local descriptor order;
distributed labels-only weights cannot describe missing cross-rank candidates.
Explicit tuples preserve their declared candidate coverage. PML's efficient
distributed mode is not an exact global-batch gradient guarantee.

## Models and named outputs

Models compute features independently of dataset names. `NamedBranches` explicitly
composes shared or separate encoders, heads and postprocessing; input routes select
which branches run. A modality need not own a tower and need not execute every head.
See [multimodal configuration](MULTIMODAL.md) for routing and stable row order.

A global output stream declares shape `[D]`; a fixed spatial stream declares
`[C,H,W]`. Published tensors retain the original sample indices. Token grids need
explicit height, width and special-token counts; variable grids are not supported
by fixed-shape storage. Output position, normalization and whitening are explicit
parts of the stream contract.

Whitening may be an offline-fitted affine or a trainable layer. Fitted artifacts
must identify their source model state, preprocessing, output space and mount
location. Validate source compatibility before assembly; arbitrary subclasses and
unverified forward implementations do not inherit a mounting guarantee.

## Training and resource ownership

`gmlab.training.runtime.fit` returns a native Lightning Trainer. Its
`trainer_options` are native options; callbacks and optimizer/scheduler factories
remain explicit components. `initialization.checkpoint` initializes a new model;
`runtime.resume` restores the training graph's checkpoint state. These modes are
mutually exclusive. Model-only weights cannot restore optimizer state.

`logger=False` disables tracking. An injected logger is borrowed; a configured
logger or default SwanLab logger is owned and finalized by the entrypoint.
Validation can follow the native schedule, require a final-step result or force
that result. Optional testing explicitly selects current, best or last weights.
The multi-source loader checkpoints its source cursor and DataLoader generator
starts. Resume re-reads and decodes the completed prefix to reproduce the
iterator's progression; it is not a seek operation. Avoiding that work requires
a sampler/worker augmentation-state contract and is not implemented here.
Deterministic, zero-worker mid-epoch replay, including gradient accumulation
and learning-rate scheduler state, is tested; worker prefetch,
random augmentation state and arbitrary custom sampler state are not guaranteed.
Other loaders retain their existing checkpoint boundaries. No per-rank RNG
restoration is implied.

## Feature extraction and precision

`gmlab.extraction.extract_features` accepts a model and native sequential
DataLoader with stable sample/preprocessing identities. It produces named stream
records; `extract_global_features` is the global descriptor convenience API.
`run_extraction` constructs these inputs from configuration. See the
[producer and reader API](artifacts/README.md#public-producer-and-reader-apis)
for runnable examples, record fields and indexed reading.

The model is moved to the requested device and left there; module training flags
are restored. Caller-owned loaders, datasets and stores are not closed or
reconfigured. Extraction does not restore generator, sampler, dataset or worker
state/cursors. Successful
cache publication and reuse persist. Features from separate datasets remain
separate immutable objects; consumers may select compatible objects together.

Offline extraction defaults to FP32. AMP and CUDA TF32 are explicit computation
policies; storage dtype only controls conversion when writing features. FP32
storage cannot make AMP computation equivalent to FP32 inference. Ordinary and
cached evaluation comparisons must use the same computation policy. Runtime
contexts restore temporary backend settings. Hardware, batch size and workers are
execution records, not independent cache-reuse barriers.

GeoDataset.geo_id(group) creates a portable namespace/group identity. Readers
must use stable source keys, not rank-local or batch-local positions. GL3D uses
scene names within identity_namespace; unrelated collections reusing those
names need distinct namespaces. University1652 uses source_scope (dataset_name
by default), class names for GeoID, and relative image paths for sample identity.
Replicas must share namespaces; unrelated sources must not share them.
GeoID is not automatically a class label. Native class-dependent losses retain
real labels and reject relation-only supervision. ClassDatasetCollection combines declared class vocabularies into one contiguous
label mapping for native metric and proxy losses.

Subgraph recipe assembly binds a read-only GT query using the collator's declared
thresholds. Each relation value carries its source and threshold identity, so a
query cannot silently replace another named GT. Explicit Python composition can
use bind_dataset_relations(criterion, loader) from gmlab.training.recipes.
Native relation mining currently supports MultiSimilarityMiner, PairMarginMiner,
TripletMarginMiner, BatchEasyHardMiner and BatchHardMiner. Other miners require
a verified mine_candidates extension; unsupported combinations fail explicitly.

ClassDatasetCollection accepts dataset instances or component specifications.
Each source must expose class_geo_ids in its local label order and return
canonical items containing rows. University1652 supplies this declaration;
scene-only datasets do not. Each source owns its transforms and sampling.
The collection preserves physical sample identities and remaps only class labels.
Its num_classes supplies the class count for native proxy losses, while
class_geo_ids records the ordered vocabulary. Source reordering does not change
labels. Adding or removing classes can change labels; restore proxy weights only
with the same vocabulary or an explicit weight remapping. Unrelated collections
of the same reader must declare different identity namespaces/source_scope.

```python
from gmlab.datasets.utils.classification import ClassDatasetCollection
from gmlab.datasets.utils.samples import stack_samples
from torch.utils.data import DataLoader

training_data = ClassDatasetCollection([first_dataset, second_dataset])
loader = DataLoader(training_data, batch_size=8, collate_fn=stack_samples)
# PML receives the resulting batch["labels"] directly.
```

Full Lightning training checkpoints persist the ordered class_geo_ids vocabulary.
Training resume rejects missing or changed vocabularies, including reorderings
with the same class count. Historical checkpoints without that metadata cannot
be resumed against a declared vocabulary; model-only initialization is a separate
operation and does not restore the criterion or optimizer state. Independent
inference without a training vocabulary does not enforce this training check.

SubgraphDatasetCollection composes scene datasets using their existing GeoIDs
and one shared query domain. Sources own their transforms and sampling; use a
normal DataLoader over the collection with PadSubgraphs. Different source
namespaces prevent coincident local scene names from colliding. Shared GeoIDs
are allowed only when node ordering and sparse GT agree. The collection queries
all registered sources even when a rank's local batch contains only one source.
All children use the collator's overlap thresholds and exclusive-scene contract.

For independent GT definitions, a Dataset may expose
relation_resolvers(**relation_options), returning queries with distinct keys.
Recipe assembly registers each query. Each RelationSupervision.source selects
its matching query; BoundObjective routes named supervision without rewriting
it. Unregistered sources and contradictory observations fail explicitly.
This does not synthesize class labels from graph GT or merge independent GTs.

When a miner is enabled, MetricObjective reports candidate and selected positive/
negative pair counts plus miner/rate. Counts use unique directed
pairs, projecting triplets onto their positive/negative edges; the rate is not a
triplet occurrence ratio. Empty candidate sets report zero. Distributed mining
reports the gathered candidate domain. Epoch aggregation follows the normal
Lightning batch-weighted scalar averaging, not a ratio of epoch-wide totals.
Additional objectives retain these reports under their objective names.

Datasets with independent class GTs can declare class_vocabularies, mapping each
supervision name to its ordered tuple of class GeoIDs. Recipe assembly records
these vocabularies in full training checkpoints and checks every declared entry
on resume. Providers must declare all class-supervised vocabularies; the runtime
does not infer class semantics from integer tensors or graph identities.

## Exact retrieval search

`gmlab.search.exact_search_chunks` serves retrieval evaluation and pair building.
Experiment YAML does not need backend, threshold, GPU preference or tie-order
settings. The shared code defaults to automatic exact search: above 4000 gallery
rows it prefers a GPU FAISS Flat index, falling back to CPU FAISS and then CPU
PyTorch when unavailable or constrained. Small galleries use PyTorch directly.
Native backend ordering is the default for equal scores. This is exhaustive
search; no approximate index is selected implicitly.
Backend selection and failures are logged. Features and index storage still
scale with gallery size; query chunks bound temporary score/result storage.
Full mAP requires all gallery ranks, so GPU FAISS large-k limits can cause CPU
fallback. Search operates on FP32 descriptors; extraction precision is separate.
Low-level Python callers can explicitly request historical PyTorch ordering or
stable index ordering for comparisons; these pin PyTorch and are not required
experiment settings. Historical result records retain their original policy.

`gmlab.retrieval.topk_neighbors` adapts self-search to COLMAP pair construction
and uses the same automatic policy. The pair CLI prefers GPU by default;
`--no-prefer-gpu` disables GPU selection. Vocabulary assignment in classical
VLAD tools owns its existing index lifecycle and is not retrieval ranking.
