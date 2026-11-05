"""Parameter selection and native optimizer/scheduler construction."""

import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from ..components import build_component
from ..config import scheduler_definition

# Python composition accepts native state or factories; YAML supplies mappings.
# Factories operate on parameters/optimizer only, never on the complete recipe.
OptimizerSpec = (
    Optimizer | Callable[[list[nn.Parameter]], Optimizer] | Mapping[str, Any]
)
SchedulerSpec = (
    LRScheduler
    | Callable[[Optimizer], LRScheduler | dict[str, Any]]
    | Mapping[str, Any]
)


def parameter_groups(module: nn.Module, groups: list[dict]) -> list[dict]:
    """Select trainable parameters by full module-name prefixes exactly once.

    Prefixes include model/criterion ownership, e.g. model.backbone. Unselected
    parameters form the default optimizer group. Overlap or empty selections
    fail rather than silently changing the learning-rate assignment.
    """
    remaining = {
        name: value for name, value in module.named_parameters() if value.requires_grad
    }
    selected_names = set()
    result = []
    for specification in groups:
        options = dict(specification)
        prefixes = options.pop("prefixes")
        if "params" in options:
            raise ValueError("Optimizer group params are selected by prefixes")
        if (
            not isinstance(prefixes, (list, tuple))
            or not prefixes
            or any(not isinstance(prefix, str) or not prefix for prefix in prefixes)
        ):
            raise ValueError("Optimizer prefixes must be a nonempty sequence of names")
        matches = {
            name: parameter
            for name, parameter in remaining.items()
            if any(
                name == prefix or name.startswith(prefix + ".") for prefix in prefixes
            )
        }
        if not matches:
            raise ValueError(
                f"Optimizer prefixes match no trainable parameters: {prefixes}"
            )
        overlap = selected_names & matches.keys()
        if overlap:
            raise ValueError(
                f"Optimizer parameter selections overlap: {sorted(overlap)}"
            )
        selected_names.update(matches)
        result.append({"params": list(matches.values()), **options})
    default = [value for name, value in remaining.items() if name not in selected_names]
    if default:
        result.append({"params": default})
    return result


def build_optimizer(module: nn.Module, specification: Mapping) -> Optimizer:
    """Build a native optimizer with optional explicit prefix-selected groups."""
    options = dict(specification)
    groups = options.pop("groups", [])
    parameters = (
        parameter_groups(module, groups)
        if groups
        else [parameter for parameter in module.parameters() if parameter.requires_grad]
    )
    return build_component(options, params=parameters)


def warmup_cosine(
    optimizer: Optimizer,
    total_steps: int,
    *,
    warmup_ratio: float = 0.1,
    warmup_start_factor: float = 0.01,
    eta_min: float = 1e-7,
) -> LRScheduler:
    """Build native step-based linear warmup followed by cosine decay.

    total_steps counts optimizer updates, including Lightning's accumulation
    and sharding rules. This preserves the existing SupScene schedule formula.
    """
    if not math.isfinite(total_steps) or total_steps < 1:
        raise ValueError("WarmupCosine requires finite positive total_steps")
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0,1)")
    warmup_steps = int(total_steps * warmup_ratio)
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=eta_min
    )
    if not warmup_steps:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def build_scheduler(
    optimizer: Optimizer,
    specification: Mapping,
    total_steps: int | Callable[[], int],
) -> dict:
    """Construct a scheduler and return native Lightning control options.

    Factories declaring total_steps receive the actual optimizer-step horizon.
    Native epoch schedulers receive only their declared parameters. New factory
    types need no branch in MetricModule or configuration validation.
    """
    constructor, arguments, controls = scheduler_definition(specification)
    if "optimizer" in arguments:
        raise ValueError("The runtime supplies scheduler.optimizer")
    if (
        "total_steps" in inspect.signature(constructor).parameters
        and "total_steps" not in arguments
    ):
        horizon = total_steps() if callable(total_steps) else total_steps
        arguments.setdefault("total_steps", horizon)
    scheduler = constructor(optimizer=optimizer, **arguments)
    return {"scheduler": scheduler, **controls}


def configure_optimization(
    module: nn.Module,
    optimizer_factory: OptimizerSpec | None,
    scheduler_factory: SchedulerSpec | None,
    total_steps: Callable[[], int],
) -> Optimizer | dict:
    """Build optimizer/scheduler state and verify exact trainable-parameter coverage.

    Python optimizer factories receive the trainable parameter list. Recipes use
    a specification with explicit prefix groups. Shared parameters appear once.
    """
    if optimizer_factory is None:
        raise ValueError("Training requires an optimizer factory")
    parameters = [
        parameter for parameter in module.parameters() if parameter.requires_grad
    ]
    if isinstance(optimizer_factory, Optimizer):
        optimizer = optimizer_factory
    elif isinstance(optimizer_factory, Mapping):
        optimizer = build_optimizer(module, optimizer_factory)
    else:
        optimizer = optimizer_factory(parameters)

    configured = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    configured_ids = [id(parameter) for parameter in configured]
    if len(configured_ids) != len(set(configured_ids)):
        raise ValueError("Optimizer contains duplicate shared parameters")
    if set(configured_ids) != {id(parameter) for parameter in parameters}:
        raise ValueError(
            "Optimizer must cover all and only trainable module parameters"
        )
    if scheduler_factory is None:
        return optimizer
    if isinstance(scheduler_factory, Mapping):
        scheduler = build_scheduler(optimizer, scheduler_factory, total_steps)
    elif hasattr(scheduler_factory, "optimizer"):
        scheduler = scheduler_factory
        if scheduler.optimizer is not optimizer:
            raise ValueError("Scheduler is bound to another optimizer")
    else:
        scheduler = scheduler_factory(optimizer)
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
