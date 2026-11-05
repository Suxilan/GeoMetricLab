"""DINOv2 facet descriptors and fixed-vocabulary VLAD, without image or cache IO."""

# Adapted from https://github.com/AnyLoc/AnyLoc (BSD 3-Clause).
# Copyright (c) 2023, AnyLoc
# See LICENSES/AnyLoc.txt and NOTICE for the retained terms and disclaimer.

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class AnyLoc(nn.Module):
    """Map normalized images [B,3,H,W] to VLAD descriptors [B,K*D].

    The caller supplies a DINOv2 backbone and vocabulary [K,D]. Geometry must
    already be a multiple of 14. No resizing, vocabulary fitting or downloading
    occurs in forward. This is an inference component (no autograd).
    """

    # Persisted calculation identifier, not an import path. The implementation
    # preserves the previous adapter's patch selection and VLAD arithmetic.
    calculation_id = "config.extraction_models.anyloc.<locals>.AnyLocModel"

    def __init__(
        self,
        backbone: nn.Module,
        centers: torch.Tensor,
        *,
        model: str = "dinov2_vitg14",
        layer: int = 31,
        facet: str = "value",
    ) -> None:
        super().__init__()
        if facet not in {"token", "query", "key", "value"}:
            raise ValueError("facet must be token, query, key or value")
        if centers.ndim != 2 or not centers.is_floating_point():
            raise ValueError("Vocabulary must be floating [K,D]")
        if not all(centers.shape) or not torch.isfinite(centers).all():
            raise ValueError("Vocabulary must be nonempty and finite")
        if not 0 <= layer < len(backbone.blocks):
            raise ValueError("layer is outside the backbone")
        self.backbone = backbone.eval()
        self.register_buffer("centers", centers)
        self.feature_config = {
            "type": "anyloc",
            "model": model,
            "layer": layer,
            "facet": facet,
            "num_clusters": len(centers),
            "use_cls": False,
            "norm_descs": True,
        }
        self._hook_output = None
        block = backbone.blocks[layer]
        target = block if facet == "token" else block.attn.qkv
        self._hook_handle = target.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._hook_output = output

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % 14 or width % 14:
            raise ValueError(
                "AnyLoc requires dimensions divisible by 14. "
                "No implicit resize, crop or padding is applied."
            )
        self._hook_output = None
        self.backbone(images)
        tokens = self._hook_output
        self._hook_output = None
        if tokens is None:
            raise RuntimeError("Selected backbone layer did not produce tokens")
        patches = tokens[:, 1:, :]
        facet = self.feature_config["facet"]
        if facet != "token":
            patches = patches.chunk(3, dim=-1)[
                {"query": 0, "key": 1, "value": 2}[facet]
            ]
        patches = F.normalize(patches, dim=-1)
        return torch.stack([vlad_descriptor(patch, self.centers) for patch in patches])


def vlad_descriptor(patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Hard cosine assignment, intra-cluster L2 then global L2; [P,D] -> [K*D]."""
    if patches.shape[-1] != centers.shape[-1]:
        raise ValueError("Patch and vocabulary dimensions differ")
    labels = (F.normalize(patches, dim=-1) @ F.normalize(centers, dim=-1).T).argmax(-1)
    residuals = patches.unsqueeze(1) - centers.unsqueeze(0)
    clusters, dimension = centers.shape
    result = patches.new_zeros(clusters * dimension)
    for cluster in range(clusters):
        mask = labels == cluster
        if mask.any():
            result[cluster * dimension : (cluster + 1) * dimension] = F.normalize(
                residuals[mask, cluster, :].sum(0), dim=0
            )
    return F.normalize(result, dim=0)


def build_anyloc(
    *,
    model="dinov2_vitg14",
    layer=31,
    facet="value",
    domain="aerial",
    num_clusters=32,
    vocabulary_dir="cache/vocabularies/anyloc",
    repository="facebookresearch/dinov2",
    source="github",
) -> AnyLoc:
    """Construct a backbone and read an existing vocabulary; never fit/download centers."""
    filename = f"{model}_l{layer}_{facet}_c{num_clusters}_{domain}_c_centers.pt"
    centers = torch.load(
        Path(vocabulary_dir) / "centers" / filename,
        map_location="cpu",
        weights_only=True,
    )
    if centers.ndim != 2 or centers.shape[0] != num_clusters:
        raise ValueError("Vocabulary shape does not match num_clusters")
    backbone = torch.hub.load(repository, model, source=source)
    return AnyLoc(backbone, centers, model=model, layer=layer, facet=facet)
