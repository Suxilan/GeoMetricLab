# Relation-supervised training

[base.yaml](base.yaml) selects `GeoPipeline` with `supervision: relations`.
Supply a relation dataset, compatible loss and a collator such as `PadSubgraphs`.

The collator pads image subgraphs and retains valid-node masks, pair masks and
sample identities. GT decoding defines which pairs are positive, negative or
unknown. Relations do not have to form classes.

## Example: GL3D

[gl3d_supscene_peft_dinov2_scpp.yaml](gl3d_supscene_peft_dinov2_scpp.yaml)
trains PEFT DINOv2 + SCPP. Prepare the [scene data](../../../datasets/README.md#gl3d)
and [DINOv2 repository](../../../third_party/README.md) first.
The recipe uses pretrained weights and one subgraph of up to 128 images per GPU.

Create `local.yaml` in this directory:

```yaml
extends: gl3d_supscene_peft_dinov2_scpp.yaml
optimizer:
  lr: 0.00002
runtime:
  run_dir: results/gl3d_local
  trainer:
    accelerator: gpu
    devices: 1
    precision: bf16-mixed
```

```bash
python -m gmlab.train --config config/gmlab/relation/local.yaml --print-config
python -m gmlab.train --config config/gmlab/relation/local.yaml
```

Set the inherited data paths for your copy. This recipe uses the held-out GL3D
test scenes for checkpoint selection; evaluating those scenes again is not an
independent test.

[u1652_global_circleloss_a6k.yaml](u1652_global_circleloss_a6k.yaml) shows the same
interface with University1652 relation GT and CircleLoss.

## Other input layouts

For modality routing, use the [multimodal base](../multimodal/README.md), set
`supervision: relations` and supply a relation collator. The built-in
`PadSubgraphs` handles homogeneous image tensors; image/point mixtures need
their own collator and batch adapter.

For several sources, use the [joint base](../joint/README.md) and set relation
supervision under `pipeline.task` or in a source's task options.
