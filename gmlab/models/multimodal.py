"""Explicit modality branches with aligned descriptor outputs."""

from collections.abc import Callable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..components import build_component, resolve_component
from .branches import _adapter_modules, grouped_input_order


def descriptor_features(output: tuple) -> torch.Tensor:
    """Select GeoEncoder's [N,D] descriptor before a modality-specific head."""
    return output[0]


class ModalityEncoder(nn.Module):
    """Encode heterogeneous modality groups and restore original row order.

    inputs contains inputs {modality: Tensor[Nm,...]} and positions {modality: [Nm]}.
    The images key remains accepted for existing image collates. Encoders define
    their own input shapes; point clouds and other Tensor inputs are valid.
    Positions must partition [0,N). Each head returns [Nm,D] or a mapping of
    named [Nm,Dk] outputs. All present modalities must agree on output names and
    width per output; different outputs may have different widths. Every output
    is restored to original row order and normalized independently if requested.
    Provide either one shared encoder or explicit per-modality encoders. Shared
    parameters are deduplicated by PyTorch parameter iteration; state_dict keeps
    ordinary module alias paths. Rebuild the same sharing before loading state.
    feature_adapter selects encoder features;
    feature_adapters can override that selection per modality. output_adapters
    select [Nm,D] from individual head outputs. Heads are ordinary modules;
    modules in adapter mappings retain their parameters in state_dict, while
    stateless selectors remain ordinary callables. Missing groups are allowed
    at inference; present groups must be nonempty. Native BatchNorm constraints
    apply per group; no implicit group merging or eval fallback occurs. Training
    with absent branches requires an explicit DDP unused-parameter/sampling policy.
    No normalization, sampling or retrieval roles are inferred from modality names.
    """

    input_layout = "grouped"

    @property
    def input_modalities(self):
        return tuple(self.heads)

    def __init__(
        self,
        heads: Mapping[str, nn.Module],
        *,
        encoder: nn.Module | None = None,
        encoders: Mapping[str, nn.Module] | None = None,
        feature_adapter: Callable | None = None,
        feature_adapters: Mapping[str, Callable] | None = None,
        output_adapters: Mapping[str, Callable] | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if (encoder is None) == (encoders is None):
            raise ValueError("Provide one shared encoder or per-modality encoders")
        if not heads:
            raise ValueError("At least one modality head is required")
        if encoders is not None and set(encoders) != set(heads):
            raise ValueError("Encoder and head modalities must match")
        self.encoder = encoder
        self.encoders = nn.ModuleDict(encoders or {})
        self.heads = nn.ModuleDict(heads)
        self.feature_adapter = feature_adapter
        self.feature_adapters = _modality_adapters(feature_adapters, heads)
        self.output_adapters = _modality_adapters(output_adapters, heads)
        self.feature_adapter_modules = _adapter_modules(self.feature_adapters, self)
        self.output_adapter_modules = _adapter_modules(self.output_adapters, self)
        if type(normalize) is not bool:
            raise TypeError("normalize must be boolean")
        if feature_adapter is not None and not callable(feature_adapter):
            raise TypeError("feature_adapter must be callable")
        self.normalize = normalize

    def forward(self, inputs: Mapping) -> torch.Tensor | dict[str, torch.Tensor]:
        order = grouped_input_order(inputs, self.heads)
        groups = inputs.get("inputs", inputs.get("images"))
        descriptors = []
        output_names = None
        named = None
        for modality, values in groups.items():
            encoder = (
                self.encoder if self.encoder is not None else self.encoders[modality]
            )
            features = encoder(values)
            feature_adapter = self.feature_adapters.get(modality, self.feature_adapter)
            if feature_adapter is not None:
                features = feature_adapter(features)
            embeddings = self.heads[modality](features)
            output_adapter = self.output_adapters.get(modality)
            if output_adapter is not None:
                embeddings = output_adapter(embeddings)
            is_named = isinstance(embeddings, Mapping)
            if named is not None and named != is_named:
                raise ValueError("Modality heads must agree on Tensor or named outputs")
            named = is_named
            outputs = dict(embeddings) if is_named else {"descriptor": embeddings}
            if not outputs or any(
                not isinstance(name, str) or not name for name in outputs
            ):
                raise ValueError("Heads require nonempty output names")
            if output_names is not None and set(outputs) != set(output_names):
                raise ValueError("Modality heads must expose the same output names")
            output_names = tuple(outputs) if output_names is None else output_names
            for name, value in outputs.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 2
                    or not value.is_floating_point()
                    or len(value) != len(values)
                ):
                    raise ValueError(
                        f"Head {modality}/{name} must return embeddings [Nm,D]"
                    )
            descriptors.append(outputs)
        result = {}
        for name in output_names:
            values = [output[name] for output in descriptors]
            if len({value.shape[1] for value in values}) != 1:
                raise ValueError(
                    "All modality heads must share one embedding dimension per output"
                )
            embeddings = torch.cat(values)
            embeddings = embeddings[order.to(embeddings.device)]
            result[name] = (
                F.normalize(embeddings, dim=1) if self.normalize else embeddings
            )
        return result if named else result["descriptor"]


def _modality_adapters(adapters: Mapping | None, heads: Mapping) -> dict:
    values = dict(adapters or {})
    if values.keys() - heads.keys() or any(
        not callable(value) for value in values.values()
    ):
        raise ValueError(
            "Modality adapters must be callable and name an existing modality"
        )
    return values


def _resolve_module(specification, shared: Mapping[str, nn.Module]) -> nn.Module:
    if isinstance(specification, Mapping) and "ref" in specification:
        if set(specification) != {"ref"} or specification["ref"] not in shared:
            raise ValueError("A local module reference requires exactly one known ref")
        return shared[specification["ref"]]
    return build_component(specification)


def _resolve_adapters(specifications: Mapping | None) -> dict:
    return {
        name: resolve_component(value)
        if isinstance(value, str)
        else build_component(value)
        for name, value in (specifications or {}).items()
    }


def build_modality_encoder(
    heads: Mapping,
    *,
    encoder: Mapping | None = None,
    encoders: Mapping | None = None,
    feature_adapter: str | Callable | None = None,
    feature_adapters: Mapping | None = None,
    output_adapters: Mapping | None = None,
    modules: Mapping | None = None,
    normalize: bool = True,
) -> ModalityEncoder:
    """Construct components; optional local modules are shared via ``{ref: name}``.

    feature_adapters override the common encoder selector per input modality.
    output_adapters select each head's [Nm,D] output. References are local to
    this composition, not a global object registry; ordinary module ownership
    and shared parameter/state_dict behavior remain PyTorch's.
    """
    shared = {name: build_component(spec) for name, spec in (modules or {}).items()}
    return ModalityEncoder(
        heads={name: _resolve_module(spec, shared) for name, spec in heads.items()},
        encoder=None if encoder is None else _resolve_module(encoder, shared),
        encoders=None
        if encoders is None
        else {name: _resolve_module(spec, shared) for name, spec in encoders.items()},
        feature_adapter=resolve_component(feature_adapter)
        if isinstance(feature_adapter, str)
        else feature_adapter,
        feature_adapters=_resolve_adapters(feature_adapters),
        output_adapters=_resolve_adapters(output_adapters),
        normalize=normalize,
    )
