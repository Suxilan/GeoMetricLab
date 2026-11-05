"""Explicit model-output and data-supervision selection for one objective."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..components import build_component
from .contracts import ContextualLoss


class BoundObjective(ContextualLoss):
    """Bind one model output and named data values to an injected objective.

    ``output`` selects a Tensor from a model-output mapping; None accepts an
    already selected Tensor. Its leading dimension denotes model input rows.
    ``arguments`` maps objective keyword names to independent supervision keys.
    Values are passed intact: labels, relations, tuple directions and identities
    are validated by their owning data/target components, never reconstructed.
    Empty arguments support differentiable feature regularizers without GT.
    Neither selection detaches or copies the differentiable model output.
    """

    def __init__(
        self,
        objective: nn.Module | Mapping[str, Any],
        *,
        output: str | None = None,
        arguments: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.objective = build_component(objective)
        if not isinstance(self.objective, nn.Module):
            raise TypeError("BoundObjective.objective must construct an nn.Module")
        if output is not None and (not isinstance(output, str) or not output):
            raise ValueError("BoundObjective.output must be a nonempty name or None")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise TypeError(
                "BoundObjective.arguments must map parameter names to GT names"
            )
        self.output = output
        self.arguments = dict(arguments or {})
        if any(
            not isinstance(parameter, str)
            or not parameter
            or not isinstance(source, str)
            or not source
            for parameter, source in self.arguments.items()
        ):
            raise ValueError(
                "BoundObjective.arguments requires nonempty parameter/GT names"
            )

    def forward(
        self,
        embeddings: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        supervision: Mapping[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return scalar loss/reports while preserving the selected output graph."""
        selected = embeddings
        if self.output is not None:
            if not isinstance(embeddings, Mapping) or self.output not in embeddings:
                raise ValueError(
                    f"BoundObjective: missing model output {self.output!r}"
                )
            selected = embeddings[self.output]
        if not torch.is_tensor(selected) or selected.ndim == 0:
            raise ValueError(
                "BoundObjective requires a Tensor [N,...]; select a named output explicitly"
            )
        if supervision is not None and not isinstance(supervision, Mapping):
            raise TypeError("BoundObjective.supervision must be a named data mapping")
        sources = supervision if supervision is not None else {}
        missing = set(self.arguments.values()) - sources.keys()
        if missing:
            raise ValueError(
                f"BoundObjective: missing supervision names {sorted(missing)}"
            )
        arguments = {
            parameter: sources[source] for parameter, source in self.arguments.items()
        }
        result = self.objective(selected, **arguments)
        if torch.is_tensor(result):
            result = {"loss": result}
        if not isinstance(result, Mapping):
            raise TypeError(
                "BoundObjective target must return a Tensor or a loss mapping"
            )
        loss = result.get("loss")
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("BoundObjective target must return a scalar Tensor loss")
        return dict(result)
