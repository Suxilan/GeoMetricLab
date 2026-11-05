"""Explicit named descriptor branches built from ordinary PyTorch modules."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ..components import build_component, resolve_component


def pooled_descriptor(output: tuple) -> torch.Tensor:
    """Select [N,D] from an aggregator's (descriptors, auxiliary) output."""
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("Expected aggregator output (descriptors, auxiliary)")
    return output[0]


def run_descriptor_head(
    features: Any,
    head: nn.Module,
    *,
    output_adapter: Callable | None = None,
    normalization: nn.Module | None = None,
    postprocess: nn.Module | None = None,
    normalize: bool = False,
    dimension: int | None = None,
) -> tuple[torch.Tensor, Any]:
    """Execute head -> adapter -> normalization -> postprocess -> optional L2.

    The selected head output is floating [N,D]. dimension, when declared, is D
    before normalization/postprocessing, not an affine's reduced output width.
    Return the descriptor and untouched head output so existing aggregators can
    retain auxiliary data. Shared BN is called separately for each invocation;
    concatenating branches would change training statistics and is not done.
    """
    raw_output = head(features)
    descriptors = (
        output_adapter(raw_output) if output_adapter is not None else raw_output
    )
    if (
        not isinstance(descriptors, torch.Tensor)
        or descriptors.ndim != 2
        or not descriptors.is_floating_point()
    ):
        raise ValueError("A descriptor head adapter must select floating [N,D]")
    if dimension is not None and descriptors.shape[1] != dimension:
        raise ValueError("Descriptor head output differs from its declared dimension")
    row_count = len(descriptors)
    if normalization is not None:
        descriptors = normalization(descriptors)
    if postprocess is not None:
        descriptors = postprocess(descriptors)
    if (
        not isinstance(descriptors, torch.Tensor)
        or descriptors.ndim != 2
        or not descriptors.is_floating_point()
        or len(descriptors) != row_count
    ):
        raise ValueError("Descriptor postprocessing must retain floating [N,D] rows")
    if normalize:
        descriptors = F.normalize(descriptors, dim=-1)
    return descriptors, raw_output


@dataclass(frozen=True)
class DescriptorBranch:
    """References into NamedBranches module pools; dimension is pre-affine D."""

    encoder: str
    head: str
    normalization: str | None = None
    postprocess: str | None = None
    normalize: bool = False
    dimension: int | None = None

    def __post_init__(self) -> None:
        for field in ("encoder", "head", "normalization", "postprocess"):
            value = getattr(self, field)
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(f"Branch {field} must be a nonempty module reference")
        if self.encoder is None or self.head is None:
            raise ValueError("Each branch needs explicit encoder and head references")
        if type(self.normalize) is not bool:
            raise ValueError("Branch normalize must be boolean")
        if self.dimension is not None and (
            type(self.dimension) is not int or self.dimension < 1
        ):
            raise ValueError("Branch dimension must be a positive integer")


def _module_pool(modules: Mapping[str, nn.Module], name: str) -> nn.ModuleDict:
    if not isinstance(modules, Mapping):
        raise TypeError(f"{name} must be a mapping of constructed modules")
    if any(not isinstance(value, nn.Module) for value in modules.values()):
        raise TypeError(f"{name} values must be constructed nn.Modules")
    return nn.ModuleDict(modules)


def _adapters(adapters: Mapping[str, Callable] | None, modules: nn.ModuleDict) -> dict:
    values = dict(adapters or {})
    if values.keys() - modules.keys() or any(
        not callable(value) for value in values.values()
    ):
        raise ValueError("Adapters must be callable and name an existing module")
    return values


def _adapter_modules(adapters: Mapping, registered: nn.Module) -> nn.ModuleDict:
    """Register callable modules once, including sharing across adapter pools."""
    existing = {id(module) for module in registered.modules()}
    modules = {}
    for name, value in adapters.items():
        if isinstance(value, nn.Module) and id(value) not in existing:
            modules[name] = value
            existing.update(id(module) for module in value.modules())
    return nn.ModuleDict(modules)


def grouped_input_order(inputs: Mapping, modalities) -> torch.Tensor:
    """Validate grouped Tensor rows before forward; return their CPU sort order.

    Each present group is nonempty. Missing modalities must be omitted, not
    represented by zero-row tensors. Positions partition [0,N) exactly.
    """
    if not isinstance(inputs, Mapping):
        raise ValueError("Grouped inputs must be a mapping")
    groups = inputs.get("inputs", inputs.get("images"))
    positions = inputs.get("positions")
    if (
        not isinstance(groups, Mapping)
        or not isinstance(positions, Mapping)
        or not groups
        or set(groups) != set(positions)
    ):
        raise ValueError("Inputs and positions need matching nonempty modalities")
    if set(groups) - set(modalities):
        raise ValueError("Unknown model modalities")
    rows = []
    for name, value in groups.items():
        index = positions[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim < 1
            or not len(value)
            or not isinstance(index, torch.Tensor)
            or index.ndim != 1
            or index.dtype != torch.long
            or len(index) != len(value)
        ):
            raise ValueError(
                "Each modality requires nonempty Tensor inputs and int64 positions [Nm]"
            )
        rows.append(index.detach().cpu())
    indices = torch.cat(rows)
    sorted_indices, order = indices.sort()
    if not torch.equal(sorted_indices, torch.arange(len(indices))):
        raise ValueError("Modality positions must partition the original rows")
    return order


class NamedBranches(nn.Module):
    """Apply named descriptor heads to explicitly shared or independent encoders.

    All encoders consume the same input. Each referenced encoder executes once,
    in first-use branch order, then its optional adapter selects features. Each
    branch executes its head, head adapter, normalization and postprocessor in
    declaration order. Module pool references make sharing explicit and avoid
    creating extra copies. feature_outputs maps output names to adapted encoder
    features, e.g. {'spatial': 'shared'}; these outputs bypass descriptor heads.

    Without modality_routes, input groups do not select branches.
    Outputs retain input row order and may have different descriptor widths.
    PyTorch deduplicates shared parameters; state_dict retains ordinary module
    pool paths. Reconstruct the same pool references before loading full state.

    With modality_routes, grouped inputs contain inputs {modality: Tensor[Nm,...]}
    (images is also accepted) and positions {modality: int64[Nm]}. Each modality
    executes only its selected
    branch; the result is one [N,D] Tensor restored to original row order. Missing
    modalities are allowed; unused parameters need an explicit DDP policy during
    training. Shared encoders/BN execute separately per group in input order,
    never by concatenating groups. Native BatchNorm sample-count constraints
    apply to each group. Feature outputs are unavailable in this mode.
    """

    @property
    def input_layout(self):
        return "grouped" if self.modality_routes else "tensor"

    @property
    def input_modalities(self):
        return tuple(self.modality_routes or {})

    @property
    def output_layout(self):
        return "descriptor" if self.modality_routes else "named"

    def __init__(
        self,
        encoders: Mapping[str, nn.Module],
        heads: Mapping[str, nn.Module],
        branches: Mapping[str, Mapping | DescriptorBranch],
        *,
        normalizations: Mapping[str, nn.Module] | None = None,
        postprocessors: Mapping[str, nn.Module] | None = None,
        encoder_adapters: Mapping[str, Callable] | None = None,
        head_adapters: Mapping[str, Callable] | None = None,
        feature_outputs: Mapping[str, str] | None = None,
        modality_routes: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.encoders = _module_pool(encoders, "encoders")
        self.heads = _module_pool(heads, "heads")
        self.normalizations = _module_pool(normalizations or {}, "normalizations")
        self.postprocessors = _module_pool(postprocessors or {}, "postprocessors")
        self.encoder_adapters = _adapters(encoder_adapters, self.encoders)
        self.head_adapters = _adapters(head_adapters, self.heads)
        if not isinstance(branches, Mapping) or not branches:
            raise ValueError("Declare at least one named descriptor branch")
        self.branches = {}
        for name, specification in branches.items():
            if not isinstance(name, str) or not name or name != name.strip():
                raise ValueError("Branch output names must be nonempty strings")
            branch = (
                specification
                if isinstance(specification, DescriptorBranch)
                else DescriptorBranch(**specification)
            )
            for field, pool in (
                ("encoder", self.encoders),
                ("head", self.heads),
                ("normalization", self.normalizations),
                ("postprocess", self.postprocessors),
            ):
                reference = getattr(branch, field)
                if reference is not None and reference not in pool:
                    raise ValueError(
                        f"Branch {name} references unknown {field}: {reference}"
                    )
            head_dimension = getattr(self.heads[branch.head], "out_channels", None)
            head_adapter = self.head_adapters.get(branch.head)
            if (
                branch.dimension is not None
                and head_dimension is not None
                and (head_adapter is None or head_adapter is pooled_descriptor)
                and branch.dimension != head_dimension
            ):
                raise ValueError(
                    f"Branch {name} dimension differs from head.out_channels"
                )
            self.branches[name] = branch
        self.modality_routes = dict(modality_routes or {})
        for modality, branch in self.modality_routes.items():
            if (
                not isinstance(modality, str)
                or not modality
                or branch not in self.branches
            ):
                raise ValueError("Each modality route must name an existing branch")
        # Functions select tensors; module adapters may also own trainable state.
        self.encoder_adapter_modules = _adapter_modules(self.encoder_adapters, self)
        self.head_adapter_modules = _adapter_modules(self.head_adapters, self)
        self.feature_outputs = dict(feature_outputs or {})
        if self.modality_routes and self.feature_outputs:
            raise ValueError(
                "Routed descriptors do not expose unaligned feature outputs"
            )
        if self.feature_outputs.keys() & self.branches.keys():
            raise ValueError("Feature and descriptor output names must be distinct")
        for name, reference in self.feature_outputs.items():
            if (
                not isinstance(name, str)
                or not name
                or name != name.strip()
                or reference not in self.encoders
            ):
                raise ValueError("Feature outputs must name an existing encoder")

    def _descriptor(self, name: str, features: Any) -> torch.Tensor:
        branch = self.branches[name]
        descriptor, _ = run_descriptor_head(
            features,
            self.heads[branch.head],
            output_adapter=self.head_adapters.get(branch.head),
            normalization=self.normalizations[branch.normalization]
            if branch.normalization is not None
            else None,
            postprocess=self.postprocessors[branch.postprocess]
            if branch.postprocess is not None
            else None,
            normalize=branch.normalize,
            dimension=branch.dimension,
        )
        return descriptor

    def _routed_forward(self, inputs: Mapping) -> torch.Tensor:
        order = grouped_input_order(inputs, self.modality_routes)
        groups = inputs.get("inputs", inputs.get("images"))
        descriptors = []
        for modality, value in groups.items():
            branch_name = self.modality_routes[modality]
            branch = self.branches[branch_name]
            features = self.encoders[branch.encoder](value)
            adapter = self.encoder_adapters.get(branch.encoder)
            if adapter is not None:
                features = adapter(features)
            descriptor = self._descriptor(branch_name, features)
            if len(descriptor) != len(value):
                raise ValueError("Routed descriptors must retain input rows")
            descriptors.append(descriptor)
        if len({value.shape[1] for value in descriptors}) != 1:
            raise ValueError("Routed descriptors must share one embedding dimension")
        output = torch.cat(descriptors)
        return output[order.to(output.device)]

    def forward(
        self, inputs: torch.Tensor | Mapping
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if self.modality_routes:
            if not isinstance(inputs, Mapping):
                raise ValueError("Routed branches require grouped inputs and positions")
            return self._routed_forward(inputs)
        if not isinstance(inputs, torch.Tensor) or inputs.ndim < 1:
            raise ValueError("NamedBranches expects an input Tensor with a batch axis")
        features = {}

        def encoded(name: str) -> Any:
            if name not in features:
                output = self.encoders[name](inputs)
                adapter = self.encoder_adapters.get(name)
                features[name] = adapter(output) if adapter is not None else output
            return features[name]

        outputs = {}
        for name, branch in self.branches.items():
            descriptor = self._descriptor(name, encoded(branch.encoder))
            if descriptor.ndim != 2 or len(descriptor) != len(inputs):
                raise ValueError(f"Branch {name} must retain input rows as [N,D]")
            outputs[name] = descriptor
        for name, reference in self.feature_outputs.items():
            value = encoded(reference)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim < 1
                or len(value) != len(inputs)
            ):
                raise ValueError(f"Feature output {name} must retain input rows")
            outputs[name] = value
        return outputs


def _build_pool(specifications: Mapping | None) -> dict[str, nn.Module]:
    return {
        name: build_component(spec) for name, spec in (specifications or {}).items()
    }


def _build_adapters(specifications: Mapping | None) -> dict[str, Callable]:
    return {
        name: resolve_component(value)
        if isinstance(value, str)
        else build_component(value)
        for name, value in (specifications or {}).items()
    }


def build_named_branches(
    encoders: Mapping,
    heads: Mapping,
    branches: Mapping,
    *,
    normalizations: Mapping | None = None,
    postprocessors: Mapping | None = None,
    encoder_adapters: Mapping | None = None,
    head_adapters: Mapping | None = None,
    feature_outputs: Mapping[str, str] | None = None,
    modality_routes: Mapping[str, str] | None = None,
) -> NamedBranches:
    """Build each local module pool once; branch references express YAML sharing."""
    return NamedBranches(
        encoders=_build_pool(encoders),
        heads=_build_pool(heads),
        branches=branches,
        normalizations=_build_pool(normalizations),
        postprocessors=_build_pool(postprocessors),
        encoder_adapters=_build_adapters(encoder_adapters),
        head_adapters=_build_adapters(head_adapters),
        feature_outputs=feature_outputs,
        modality_routes=modality_routes,
    )
