"""Factories and capabilities for native PML losses and custom extensions."""

import importlib
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from typing import Any

from pytorch_metric_learning import distances, losses, reducers
from torch import nn

from ..components import build_component, resolve_component
from .contracts import ContextualLoss, LossInputContract
from .pml.weighted_supcon import WeightedSupConLoss

# Verified native implementations describe inputs, not loss/miner combinations.
# Unchanged native entry points retain these defaults. Other extensions declare
# their own fields on the module/registration instead of editing this table.
_PAIR_INPUTS = LossInputContract(
    requires_labels=False, tuple_usage="pairs", supports_ref_emb=True
)
_TRIPLET_INPUTS = LossInputContract(
    requires_labels=False, tuple_usage="triplets", supports_ref_emb=True
)
_WEIGHT_INPUTS = LossInputContract(
    requires_labels=True, tuple_usage="sample_weights", supports_ref_emb=False
)
_NO_TUPLE_INPUTS = LossInputContract(
    requires_labels=True, tuple_usage="unsupported", supports_ref_emb=False
)
_NATIVE_INPUTS = {
    component: _PAIR_INPUTS
    for component in (
        losses.CircleLoss,
        losses.ContrastiveLoss,
        losses.IntraPairVarianceLoss,
        losses.GeneralizedLiftedStructureLoss,
        losses.LiftedStructureLoss,
        losses.MultiSimilarityLoss,
        losses.NTXentLoss,
        losses.SignalToNoiseRatioContrastiveLoss,
        losses.SupConLoss,
        losses.TupletMarginLoss,
    )
}
_NATIVE_INPUTS.update(
    {
        losses.TripletMarginLoss: _TRIPLET_INPUTS,
        losses.MarginLoss: _TRIPLET_INPUTS,
        losses.CosFaceLoss: _WEIGHT_INPUTS,
        losses.ArcFaceLoss: _WEIGHT_INPUTS,
        losses.LargeMarginSoftmaxLoss: _WEIGHT_INPUTS,
        losses.SphereFaceLoss: _WEIGHT_INPUTS,
        losses.ProxyAnchorLoss: _WEIGHT_INPUTS,
        losses.ProxyNCALoss: _WEIGHT_INPUTS,
        losses.SoftTripleLoss: _WEIGHT_INPUTS,
        losses.FastAPLoss: _WEIGHT_INPUTS,
        losses.P2SGradLoss: _NO_TUPLE_INPUTS,
        losses.RankedListLoss: _NO_TUPLE_INPUTS,
    }
)
# Retained for diagnostics enumerating the established native relation coverage.
# Runtime resolution uses actual types and declarations, never this name set.
RELATION_LOSSES = {
    t.__name__ for t, contract in _NATIVE_INPUTS.items() if contract.supports_relations
}

ALIASES = {
    "TripletLoss": "TripletMarginLoss",
    "LiftedLoss": "GeneralizedLiftedStructureLoss",
}

CUSTOM_LOSSES: dict[
    str, tuple[Callable, bool | None, LossInputContract | Mapping | None]
] = {
    "WeightedSupConLoss": (WeightedSupConLoss, None, WeightedSupConLoss.input_contract),
}


def build_pml_component(spec, namespace, kwargs=None):
    """Construct a native name, registered/path specification, or module instance.

    Only distance and reducer arguments are nested components here. Ordinary
    dictionaries remain constructor data; this is not recursive config guessing.
    """
    options = dict(kwargs or {})
    if isinstance(spec, nn.Module):
        if options:
            raise ValueError(
                "An injected module cannot also have constructor arguments"
            )
        return spec
    if isinstance(spec, Mapping):
        if options:
            raise ValueError("Put constructor arguments in the component specification")
        options = dict(spec)
        if "type" not in options:
            raise ValueError("PML components require an explicit type")
        spec = options.pop("type")
    for key, owner in (("distance", distances), ("reducer", reducers)):
        if key in options and options[key] is not None:
            options[key] = build_pml_component(options[key], owner)
    factory = getattr(namespace, spec, None) if isinstance(spec, str) else spec
    if factory is None:
        return build_component({"type": spec, **options})
    return factory(**options)


def register_loss(
    name: str,
    factory: Callable[..., nn.Module],
    *,
    supports_relations: bool | None = None,
    input_contract: LossInputContract | Mapping[str, Any] | None = None,
) -> None:
    """Register a PML-compatible factory without replacing an existing name.

    An explicit boolean declares support for pair/triplet supervision without
    class labels. None defers to the constructed module, then native capabilities.
    Conflicting declarations are rejected; this does not declare DDP equivalence.
    input_contract accepts a LossInputContract or field mapping. It is merged
    with the constructed module's declaration, not inferred from the factory name.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Loss registration name must be a nonempty string")
    if not callable(factory):
        raise TypeError("Loss factory must be callable")
    if hasattr(losses, name) or name in CUSTOM_LOSSES or name in ALIASES:
        raise ValueError(f"Loss already registered: {name}")
    CUSTOM_LOSSES[name] = (factory, supports_relations, input_contract)


def build_loss(
    name: str | nn.Module | Mapping[str, Any],
    kwargs: Mapping[str, Any],
    custom_module: str | None,
    custom_supports_relations: bool | None,
) -> tuple[nn.Module, str, LossInputContract]:
    """Return (loss module, canonical name, resolved input contract).

    Class-dependent objectives may require labels even when tuples are present.
    Custom implementations declare input semantics without decoding any GT here.
    """
    registered_inputs = None
    if isinstance(name, Mapping):
        if kwargs or custom_module:
            raise ValueError(
                "Put loss constructor arguments in its component specification"
            )
        kwargs = dict(name)
        if "type" not in kwargs:
            raise ValueError("Loss components require an explicit type")
        name = kwargs.pop("type")
    if isinstance(name, nn.Module):
        if custom_module:
            raise ValueError("An injected loss cannot also specify custom_loss_module")
        loss = build_pml_component(name, losses, kwargs)
        canonical_name = type(loss).__name__
        declared = None
    else:
        canonical_name = ALIASES.get(name, name)
        declared = None
        if custom_module:
            factory = getattr(importlib.import_module(custom_module), canonical_name)
        elif canonical_name in CUSTOM_LOSSES:
            factory, declared, registered_inputs = CUSTOM_LOSSES[canonical_name]
        else:
            factory = getattr(losses, canonical_name, None)
            if factory is None:
                factory = resolve_component(canonical_name)
        loss = build_pml_component(factory, losses, kwargs)
    if not isinstance(loss, nn.Module):
        raise TypeError("A loss factory must return an nn.Module")
    try:
        contract = resolve_loss_inputs(
            loss,
            registered=registered_inputs,
            registered_relations=declared,
            requested_relations=custom_supports_relations,
        )
    except (TypeError, ValueError) as error:
        error_type = TypeError if isinstance(error, TypeError) else ValueError
        raise error_type(f"loss_fn_name={canonical_name}: {error}") from error
    return loss, canonical_name, contract


def _input_declaration(value, source):
    if value is None:
        return LossInputContract()
    if isinstance(value, Mapping):
        try:
            return LossInputContract(**value)
        except (TypeError, ValueError) as error:
            error_type = TypeError if isinstance(error, TypeError) else ValueError
            raise error_type(f"{source}.input_contract: {error}") from error
    if not isinstance(value, LossInputContract):
        raise TypeError(
            f"{source}.input_contract must be a LossInputContract or mapping"
        )
    return value


def _native_input_contract(loss):
    """Retain proven native input handling when a subclass inherits it unchanged.

    Compare both forward and compute_loss, not class names or inheritance alone.
    Overridden entry points require declarations or retain runtime validation.
    This infers no reducer/local-row behavior from the native input lifecycle.
    """
    actual = type(loss)
    if actual in _NATIVE_INPUTS:
        return _NATIVE_INPUTS[actual]
    inherited = {
        contract
        for native, contract in _NATIVE_INPUTS.items()
        if getattr(actual, "forward", None) is native.forward
        and getattr(actual, "compute_loss", None) is native.compute_loss
    }
    return inherited.pop() if len(inherited) == 1 else LossInputContract()


def resolve_loss_inputs(
    loss, *, registered=None, registered_relations=None, requested_relations=None
) -> LossInputContract:
    """Resolve explicit declarations after construction, then verified defaults.

    Module/registration fields must agree. Legacy relation declarations remain
    accepted but cannot contradict explicit inputs or relax proven native
    requirements. Unknown components keep unknown fields for runtime validation.
    """
    target = type(loss).__name__
    declarations = {
        "module": _input_declaration(getattr(loss, "input_contract", None), target),
        "registration": _input_declaration(registered, target + " registration"),
    }
    explicit = {}
    for field in fields(LossInputContract):
        supplied = {
            source: getattr(contract, field.name)
            for source, contract in declarations.items()
            if getattr(contract, field.name) is not None
        }
        if len(set(supplied.values())) > 1:
            raise ValueError(
                f"{target}: Conflicting input_contract.{field.name}: {supplied}"
            )
        if supplied:
            explicit[field.name] = next(iter(supplied.values()))

    relation_declarations = {
        "module": getattr(loss, "supports_relations", None),
        "registration": registered_relations,
        "caller": requested_relations,
    }
    supplied = {k: v for k, v in relation_declarations.items() if v is not None}
    if any(type(value) is not bool for value in supplied.values()):
        raise TypeError(
            f"{target}: supports_relations declarations must be bool or None"
        )
    if len(set(supplied.values())) > 1:
        raise ValueError(
            f"{target}: Conflicting supports_relations declarations: {supplied}"
        )
    legacy_relation = next(iter(supplied.values()), None)
    native = _native_input_contract(loss)
    class_beta = (
        isinstance(loss, losses.MarginLoss)
        and type(loss).forward is losses.MarginLoss.forward
        and type(loss).compute_loss is losses.MarginLoss.compute_loss
        and loss.beta.numel() > 1
    )
    if class_beta:
        native = replace(_TRIPLET_INPUTS, requires_labels=True)
        if legacy_relation is True:
            raise ValueError(
                "Class-dependent MarginLoss cannot declare supports_relations=True"
            )
    if legacy_relation is True:
        if explicit.get("requires_labels") is True:
            raise ValueError(
                f"{target}: Conflicting supports_relations and input_contract.requires_labels"
            )
        if explicit.get("tuple_usage") == "unsupported":
            raise ValueError(
                f"{target}: Conflicting supports_relations and unsupported tuple input"
            )
        explicit["requires_labels"] = False
        if native.tuple_usage is None and "tuple_usage" not in explicit:
            explicit["tuple_usage"] = "native"
    elif legacy_relation is False:
        candidate = replace(native, **explicit)
        if "requires_labels" in explicit and candidate.supports_relations:
            raise ValueError(
                f"{target}: Conflicting supports_relations and input_contract"
            )
        if candidate.supports_relations:
            # A legacy false declaration narrows verified optional-label input;
            # it does not prove that an unknown component always needs labels.
            explicit["requires_labels"] = True
    contract = replace(native, **explicit)
    if native.requires_labels is True and contract.requires_labels is False:
        raise ValueError(
            f"{target}: requires class IDs; the declared contract cannot remove them"
        )
    if native.supports_ref_emb is False and contract.supports_ref_emb is True:
        raise ValueError(f"{target}: native ref_emb is unsupported")
    if native.tuple_usage is not None and contract.tuple_usage not in (
        native.tuple_usage,
        "unsupported",
    ):
        raise ValueError(f"{target}: native tuple usage is {native.tuple_usage}")
    return contract


def instantiate_objective(
    spec: ContextualLoss | Mapping[str, Any], *, location: str = "objective"
) -> ContextualLoss:
    """Resolve an injected contextual objective or explicit component specification."""
    if isinstance(spec, ContextualLoss):
        return spec
    try:
        objective = build_component(spec)
        if not isinstance(objective, ContextualLoss):
            raise TypeError("Composed objectives must implement ContextualLoss")
    except (TypeError, ValueError) as error:
        error_type = TypeError if isinstance(error, TypeError) else ValueError
        raise error_type(f"{location}: {error}") from error
    return objective
