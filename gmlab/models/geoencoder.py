"""Image encoder composition, independent of supervision and training runtime."""

from collections.abc import Callable, Mapping
import inspect

import torch
from torch import nn

from ..components import build_component, component_definition, resolve_component
from .branches import pooled_descriptor, run_descriptor_head


def patch_tokens(output: tuple) -> torch.Tensor:
    """Select DINO's patch map [N,C,H,W] from (CLS, patches, attention)."""
    if not isinstance(output, tuple) or len(output) != 3:
        raise ValueError("Expected the DINO (CLS, patches, attention) tuple")
    return output[1]


def _build_aggregator(specification, backbone: nn.Module) -> nn.Module:
    """Inject channels only into constructors explicitly declaring in_channels.

    Other heads must be fully configured and expose out_channels. An already
    constructed module retains its own configuration and parameter ownership.
    """
    if not isinstance(specification, Mapping):
        return build_component(specification)
    constructor, _ = component_definition(specification)
    parameters = inspect.signature(constructor).parameters
    injected = {}
    if "in_channels" in parameters:
        injected["in_channels"] = backbone.out_channels
    return build_component(specification, **injected)


class GeoEncoder(nn.Module):
    """Encode float images [N,3,H,W] as (descriptors [N,D], auxiliary dict).

    Callers may inject a backbone exposing
    ``out_channels`` and an aggregator returning ``(descriptors, auxiliary)``.
    BN, affine whitening and final L2 normalization follow the old encoder's
    order. Whitening here is a trainable Linear layer, not a fitted PCA claim.
    Pretrained ResNet construction may access timm's weight cache/network.
    No input normalization or image resizing is performed by this model.
    """

    input_layout = "tensor"
    output_layout = "descriptor"

    def __init__(
        self,
        backbone: nn.Module,
        aggregator: nn.Module | None = None,
        *,
        whitening: bool = False,
        whitening_dim: int = 256,
        whitening_trainable: bool = True,
        final_norm: bool = True,
        use_bn: bool = True,
        feature_adapter: Callable | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("GeoEncoder receives a constructed backbone nn.Module")
        if aggregator is None:
            from .aggregator.gem import GeM

            aggregator = GeM(backbone.out_channels)
        if not isinstance(aggregator, nn.Module):
            raise TypeError("GeoEncoder receives a constructed aggregator nn.Module")
        self.backbone = backbone
        self.aggregator = aggregator
        channels = aggregator.out_channels
        self.bn = nn.BatchNorm1d(channels) if use_bn else nn.Identity()
        if whitening and not 0 < whitening_dim <= channels:
            raise ValueError(
                "whitening_dim must be positive and <= aggregator channels"
            )
        self.whitening_layer = (
            nn.Linear(channels, whitening_dim) if whitening else nn.Identity()
        )
        if type(whitening_trainable) is not bool:
            raise TypeError("whitening_trainable must be boolean")
        self.whitening_layer.requires_grad_(whitening_trainable)
        self.out_channels = whitening_dim if whitening else channels
        self.final_norm = final_norm
        if feature_adapter is not None and not callable(feature_adapter):
            raise TypeError("feature_adapter must be a callable")
        self.feature_adapter = feature_adapter

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, dict]:
        features = self.backbone(images)
        if self.feature_adapter is not None:
            features = self.feature_adapter(features)
        descriptors, raw_output = run_descriptor_head(
            features,
            self.aggregator,
            output_adapter=pooled_descriptor,
            normalization=self.bn,
            postprocess=self.whitening_layer,
            normalize=self.final_norm,
        )
        return descriptors, raw_output[1]


def build_geoencoder(
    backbone: Mapping,
    aggregator: Mapping | None = None,
    *,
    whitening: bool = False,
    whitening_dim: int = 256,
    whitening_trainable: bool = True,
    final_norm: bool = True,
    use_bn: bool = True,
    feature_adapter: str | Callable | None = None,
) -> GeoEncoder:
    """Construct the YAML encoder from explicit component specifications.

    Channel injection and callable lookup belong to assembly. GeoEncoder itself
    receives only modules and callables and performs the descriptor computation.
    """
    backbone_module = build_component(backbone)
    aggregator_module = (
        _build_aggregator(aggregator, backbone_module)
        if aggregator is not None
        else None
    )
    if isinstance(feature_adapter, str):
        feature_adapter = resolve_component(feature_adapter)
    return GeoEncoder(
        backbone=backbone_module,
        aggregator=aggregator_module,
        whitening=whitening,
        whitening_dim=whitening_dim,
        whitening_trainable=whitening_trainable,
        final_norm=final_norm,
        use_bn=use_bn,
        feature_adapter=feature_adapter,
    )


def spatial_features(output: torch.Tensor, batch_info: Mapping) -> torch.Tensor:
    """Select a backbone's actual [N,C,H,W] output for common extraction.

    batch_info is the extraction adapter context; no grid is inferred and no
    descriptor normalization or patch sampling is applied.
    """
    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        raise ValueError("Expected backbone spatial features [N,C,H,W]")
    return output


def dino_patch_features(output: tuple, batch_info: Mapping) -> torch.Tensor:
    """Select DINO's explicit patch map for the two-argument extraction adapter."""
    return spatial_features(patch_tokens(output), batch_info)


def prepare_whitening_encoder(
    recipe: Mapping, calculation: Mapping, *, model: nn.Module, dimension: int
) -> dict:
    """Prepare the legacy single-descriptor recipe through common affine assembly."""
    from .assembly import prepare_whitening

    return prepare_whitening(
        recipe,
        {"descriptor": calculation},
        model=model,
        dimension=dimension,
        assignments={"descriptor": "whitening"},
        trainable=True,
    )


def attach_whitening(
    model: nn.Module, calculation: Mapping, weight: torch.Tensor, bias: torch.Tensor
) -> GeoEncoder:
    """Attach post-BN affine and final L2, preserving legacy parameter names."""
    from .assembly import attach_whitening as attach

    return attach(
        model,
        {"descriptor": calculation},
        {"whitening": {"weight": weight, "bias": bias}},
        assignments={"descriptor": "whitening"},
        trainable=True,
    )
