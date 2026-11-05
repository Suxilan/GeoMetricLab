"""Composition and explicit argument routing for metric objectives."""

import math
from collections.abc import Mapping

import torch
from torch import nn

from .contracts import ContextualLoss
from .objectives import PMLObjective
from .registry import instantiate_objective


class GeoMetricLoss(ContextualLoss):
    """Combine a primary objective, additional main objectives and auxiliaries.

    Each objective returns a scalar ``loss``. ``main_loss`` includes the primary
    and additional weights; ``aux_loss`` includes auxiliary weights. The final
    ``loss`` is their sum. This module performs no GT conversion or mining.
    Targets specific to an objective belong in objective_kwargs[name]; the main
    objective is named "primary". Shared labels/indices are standard PML inputs.
    Objectives with supervision_keys receive only those same-name fields from
    supervision; BoundObjective handles explicit output/GT remapping.
    """

    def __init__(
        self,
        *,
        primary_objective=None,
        loss_weight: float = 1.0,
        additional_losses: Mapping | None = None,
        auxiliaries: Mapping | None = None,
    ) -> None:
        super().__init__()
        self.loss_weight = self._validate_weight(loss_weight)
        self.primary_objective = (
            PMLObjective()
            if primary_objective is None
            else instantiate_objective(
                primary_objective, location="GeoMetricLoss.primary_objective"
            )
        )
        self.additional_losses, self.additional_weights = self._build_components(
            additional_losses
        )
        self.auxiliaries, self.auxiliary_weights = self._build_components(auxiliaries)
        names = set(self.additional_losses) | set(self.auxiliaries)
        if "primary" in names or set(self.additional_losses) & set(self.auxiliaries):
            raise ValueError("Objective names must be unique; primary is reserved")

    def forward(
        self,
        embeddings,
        labels: torch.Tensor | None = None,
        *,
        indices_tuple=None,
        ref_emb=None,
        objective_kwargs: Mapping | None = None,
        supervision: Mapping | None = None,
    ) -> dict[str, torch.Tensor]:
        """Route explicitly supplied inputs and return weighted scalar reports.

        Native PML inputs use embeddings [A,D], labels [A], pair/triplet indices
        and optional references [R,D]. Injected head objectives may accept named
        embeddings and supervision mappings. Each objective owns its contract.
        objective_kwargs permits distinct targets without leaking auxiliary GT
        into other objectives. Unknown objective names and conflicting inputs fail.
        """
        if isinstance(labels, Mapping):
            raise ValueError(
                "labels contains class IDs; use supervision for named targets"
            )
        routed = dict(objective_kwargs or {})
        valid_names = {"primary", *self.additional_losses, *self.auxiliaries}
        if routed.keys() - valid_names:
            raise ValueError(
                f"Unknown objective targets: {routed.keys() - valid_names}"
            )
        shared = {
            key: value
            for key, value in {
                "labels": labels,
                "indices_tuple": indices_tuple,
                "ref_emb": ref_emb,
                "supervision": supervision,
            }.items()
            if value is not None
        }

        def execute(name, objective):
            arguments = dict(routed.get(name, {}))
            if arguments.keys() & shared.keys():
                raise ValueError(f"Conflicting shared and objective inputs for {name}")
            inputs = dict(shared)
            keys = getattr(objective, "supervision_keys", None)
            if supervision is not None and keys is not None:
                inputs.pop("supervision")
                selected = {key: supervision[key] for key in keys if key in supervision}
                if selected.keys() & (inputs.keys() | arguments.keys()):
                    raise ValueError(f"Conflicting supervision inputs for {name}")
                inputs.update(selected)
            output = objective(embeddings, **inputs, **arguments)
            self._loss_value(output, name)
            return dict(output)

        result = execute("primary", self.primary_objective)
        primary_loss = result["loss"]
        main = self.loss_weight * primary_loss
        auxiliary = primary_loss.new_zeros(())
        for prefix, modules, weights in (
            ("additional", self.additional_losses, self.additional_weights),
            ("auxiliary", self.auxiliaries, self.auxiliary_weights),
        ):
            for name, objective in modules.items():
                if weights[name] == 0:
                    continue
                reports = execute(name, objective)
                value = reports["loss"]
                result[f"{prefix}/{name}"] = value
                for key, report in reports.items():
                    if key != "loss":
                        result[f"{prefix}/{name}/{key}"] = report
                if prefix == "additional":
                    main = main + weights[name] * value
                else:
                    auxiliary = auxiliary + weights[name] * value
        result.update(loss=main + auxiliary, main_loss=main, aux_loss=auxiliary)
        return result

    @staticmethod
    def _validate_weight(weight: float) -> float:
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError("Objective weights must be finite")
        return weight

    @classmethod
    def _build_components(
        cls, specs: Mapping | None
    ) -> tuple[nn.ModuleDict, dict[str, float]]:
        modules = nn.ModuleDict()
        weights = {}
        for name, spec in (specs or {}).items():
            options = dict(spec)
            weights[name] = cls._validate_weight(options.pop("weight"))
            component_spec = "type" in options or {"module", "class"} <= options.keys()
            if "objective" in options and not component_spec:
                objective = options.pop("objective")
                if options:
                    raise ValueError(
                        f"GeoMetricLoss.objectives[{name}] cannot mix an objective "
                        f"wrapper with component arguments: {sorted(options)}"
                    )
            else:
                objective = options
            modules[name] = instantiate_objective(
                objective, location=f"GeoMetricLoss.objectives[{name}]"
            )
        return modules, weights

    @staticmethod
    def _loss_value(output: Mapping, name: str) -> torch.Tensor:
        """Validate the differentiable scalar shared by all objective roles."""
        if not isinstance(output, Mapping):
            raise TypeError(f"{name} must return a mapping containing loss")
        value = output.get("loss")
        if not torch.is_tensor(value) or value.ndim != 0:
            raise ValueError(f"{name} must return a scalar Tensor loss")
        return value
