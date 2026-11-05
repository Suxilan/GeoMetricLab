# Global descriptor retrieval and reranking

The common path is:

```text
live model descriptors ─┐
                       ├─ query/gallery [Q,D]/[G,D] ─ refinement ─ exact search ─ ranked gallery positions
verified cache reader ─┘                                                          └─ optional GT metrics
```

Reranking consumes global descriptors. It does not run local matching, geometric
verification, feature extraction, or GT decoding. The selected query/gallery
descriptors must inhabit a compatible learned space; equal width alone does not
prove compatibility. Multi-modal branches may differ in architecture/preprocessing
while intentionally learning the same space. The experiment owns that pairing.

## Python: live or cached descriptors

```python
from gmlab.retrieval import retrieve

# query_features/gallery_features: finite float [Q,D]/[G,D], on the same device.
# They may come directly from model inference; no cache round-trip is required.
indices = retrieve(
    query_features,
    gallery_features,
    k=20,
    reranking={"method": "aqe", "k": 10, "alpha": 3.0, "gamma": 0.5},
    metric="cosine",
    backend="auto",
    tie_break="native",
)
# CPU int64 [Q,min(20,G)], in best-first order; map to your stable gallery IDs.
matches = [[gallery_ids[i] for i in row] for row in indices.tolist()]
```

`reranking=None` performs ordinary retrieval. The algorithm's `k` controls its
neighborhood; retrieval's `k` controls returned result count. `retrieve_chunks`
yields `(query_offset, indices)` to bound result memory. Refinement runs once on
the complete inputs before search chunking. These functions do not infer identity
or remove self matches. The evaluation protocol and pair exporter own that policy.

For offline features, replace only the input acquisition:

```python
from gmlab.datasets.utils.feature_io import read_global_features

query = read_global_features(record="results/query.json", stream="global")
gallery = read_global_features(record="results/gallery.json", stream="global")
indices = retrieve(
    query["features"],
    gallery["features"],
    k=20,
    reranking={"method": "aqe", "k": 10},
)
names = [[gallery["image_names"][i] for i in row] for row in indices.tolist()]
```

Use the actual stream names in your extraction records; omit `stream` for an
unambiguous single-stream record. Readers verify payloads and row identities.
Refined descriptors are transient; original cache contents are never overwritten.
Compare live/cached results with identical model state, preprocessing, selected
output, forward precision and query/gallery populations. Float32 storage does not
imply FP32 model computation. Search/refinement operates in FP32.

## Command line: export actual rankings

Create a retrieval options file such as `retrieval.yaml`:

```yaml
search:
  metric: cosine
  backend: auto
  prefer_gpu: true
  tie_break: native
reranking:
  method: aqe
  k: 10
  alpha: 3.0
  gamma: 0.5
```

```bash
python tools/retrieval/retrieve.py \
  --query-record results/query.json --gallery-record results/gallery.json \
  --config retrieval.yaml --topk 20 --output results/rankings.jsonl
```

The first JSONL line records verified input references and configuration. Following
lines contain `query_index`, `query_name`, `gallery_indices` and `gallery_names`
in ranked order. No GT is required. Add `--query-stream`/`--gallery-stream` for
multi-stream records. Tools run from a source checkout; installed Python callers
use `gmlab.retrieval.retrieve` from any directory.

For single-collection SfM pairs, the existing exporter accepts
`--reranking-config rerank.yaml`, where that file contains only the algorithm
mapping (`method`, `k`, etc.). It removes self pairs and deduplicates undirected
pairs **after** ranking. This exported pair set is not a query-level ranking list.

## Training validation and standalone evaluation

Add the same algorithm mapping to a training/evaluation recipe:

Task bases bind the input and output through `GeoPipeline`/`MultimodalPipeline`.
For a named multi-head model, `pipeline.retrieval_output` selects the common
descriptor space used by both live evaluation and cache extraction.

```yaml
evaluation:
  metric: cosine
  k_values: [1, 10, 25]
  full_ranking: true
  reranking: {method: aqe, k: 10, alpha: 3.0, gamma: 0.5}
```

`data.valid.sources.<name>.evaluation` can override this per source. Both native
Lightning validation and `python -m gmlab.evaluate` call `evaluate_retrieval`, which
uses the same `retrieve_chunks` path. It applies the Dataset's query/gallery rows,
self policy and GT to rankings; reranking never rewrites GT. Without GT, use
`retrieve` instead of the metric evaluator. Cached features can be passed to
`evaluate_retrieval` with the same manifest indices and protocol.

## What the included algorithms actually do

These are the project's migrated **descriptor refinement implementations**.
Names do not imply numerical equivalence to every paper implementation with the
same acronym. All use global features; no local keypoint reranker is included.

| Method | Operation | Population dependency |
| --- | --- | --- |
| `aqe` | Softmax-weighted gallery neighbors expand each query | Queries are independent for a fixed gallery |
| `dba` | Gallery neighbors augment gallery descriptors, excluding self by row ID | Gallery-only transformation |
| `dfs` | Fixed-neighborhood iterative smoothing | Joint query + gallery graph |
| `krnn` | Reciprocal-neighbor weighted aggregation and residual fusion | Joint query + gallery graph; not Jaccard distance reranking |
| `gnn` | Two graph propagation steps produce graph descriptors | Joint graph, output width Q+G, quadratic host/device memory |

For `aqe`/`dba`/`dfs`/`krnn`, cosine affinities or negative squared L2 affinities
weight closer neighbors more strongly. All normalize the input features used for
neighbor construction. This differs from raw, unnormalized Euclidean retrieval.
`gnn` uses cosine neighborhoods. Algorithm parameters are explicit keyword args;
defaults preserve historical values, not universal recommendations.

Changing the supplied query set can change DFS/KRNN/GNN gallery embeddings and
rankings. Do not rerank independent minibatches and call the result equivalent to
whole-collection evaluation. For a low-latency service, AQE supports each incoming
query; DBA's gallery transformation can be prepared externally for a fixed gallery.
This library currently has no persistent fitted reranker/index lifecycle.

Query/gallery are separate row populations. If the same physical image appears
in both roles, joint methods currently include both rows in their graph; self
exclusion at result evaluation does not deduplicate that graph. GNN chunking bounds
propagation workspace, not quadratic adjacency storage. Full mAP also requires
full rankings; Top-K output alone does not establish full mAP.

External Python refiners may be injected as callables returning `(queries, gallery)`.
They must preserve row order/count and device, return finite floating matrices,
and agree on output width. The API validates shapes/counts/devices but cannot prove
semantic row order. They consume no labels or GT.
