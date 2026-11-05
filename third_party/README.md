# Third-party repositories

The source release registers three model repositories as Git submodules.
The other local directories listed below are ignored reference or experiment
checkouts. None of these full repositories is bundled with the Python package.
Adaptations copied into `gmlab` are distributed with their own notices and license
texts; see [NOTICE](../NOTICE).

## Runtime submodules

| Directory | Commit | Consumer | Upstream terms |
| --- | --- | --- | --- |
| `dinov2/` | `b8931f7` | DINOv2 and PEFT DINOv2 backbones | [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/b8931f7bf91576930313be2c6d6af376033b35f0/LICENSE) |
| `dinov3/` | `54694f7` | DINOv3, PEFT DINOv3 and ConvNeXt backbones | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/54694f7627fd815f62a5dcc82944ffa6153bbb76/LICENSE.md) |
| `Depth-Anything-V2/` | `e5a2732` | DepthAnythingV2Backbone | [Apache-2.0 code; Small model Apache-2.0, Base/Large/Giant models CC BY-NC 4.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1/README.md#license) |

Initialize the repositories needed by your model, or fetch all three:

```bash
git submodule update --init --recursive
```

Set the backbone's `repository` field to the relevant directory. Use an absolute
path when running outside this checkout. Pretrained weights are separate inputs.

For a source archive without Git metadata, clone the required upstream repository
separately, check out the commit above and pass its path as `repository`.

## Local references and experiments

These eleven directories are not included in the release or initialized by
`git submodule update`. A local Git checkout is not automatically a submodule
of GeoMetricLab. The maintained runtime does not import these directories.

| Directory / upstream | Local copy | Purpose | Code license |
| --- | --- | --- | --- |
| [AnyLoc](https://github.com/AnyLoc/AnyLoc) | Git `a9fda68` | Reference for `gmlab.models.anyloc`; vocabularies are separate assets | BSD-3-Clause |
| [Boq](https://github.com/amaralibey/Bag-of-Queries) | Snapshot; no Git metadata | Reference for the BoQ aggregator and ResNet adapter | MIT |
| [CosPlace](https://github.com/gmberton/CosPlace) | Snapshot; no Git metadata | Reference for the CosPlace aggregator | MIT |
| [CricaVPR](https://github.com/Lu-Feng/CricaVPR) | Git `f53e941` | Reference for the CricaVPR aggregator | MIT |
| [salad](https://github.com/serizba/salad) | Snapshot; no Git metadata | Reference for the SALAD aggregator | GPLv3 |
| [Pi3](https://github.com/yyfz/Pi3) | Git `9fa3ddb` | Local geometry and overlap-GT generation | BSD-3-Clause; weights CC BY-NC 4.0 |
| [Revisit-Anything](https://github.com/AnyLoc/Revisit-Anything) | Git `628508d` | Historical VPR data organization and readers | BSD-3-Clause; bundled components retain their own terms |
| [VGGT](https://github.com/facebookresearch/vggt) | Git `44b3afb` | Local geometry experiments and architecture reference | VGGT License; checkpoint-specific terms |
| [dust3r](https://github.com/naver/dust3r) | Git `4c24a6e` | Local reconstruction and matching experiments | CC BY-NC-SA 4.0 |
| [mast3r](https://github.com/naver/mast3r) | Git `f5209af` | Local reconstruction and matching experiments | CC BY-NC-SA 4.0 |
| [vismatch](https://github.com/gmberton/vismatch) | Git `2611c3b` | Historical pair-matching backend | BSD-3-Clause wrapper; each matcher has its own terms |

Weight terms must be checked separately. In the listed VGGT checkout, only
[VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial) permits
commercial use; the original checkpoint remains non-commercial.
[DUSt3R](https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/README.md#checkpoints)
and [MASt3R](https://github.com/naver/mast3r/blob/f5209afc300cec36239a7ac992263f36847bbba0/CHECKPOINTS_NOTICE)
also specify training-data or base-checkpoint terms. These assets are not shipped
with GeoMetricLab. COLMAP is an external executable, not a release submodule.
