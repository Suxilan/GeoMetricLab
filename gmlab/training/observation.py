"""Opt-in tensor observations using native module hooks and Lightning loggers."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn
from torch.distributed import ReduceOp
from torch.utils.hooks import RemovableHandle


def _validate_tensor(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0 or value.is_complex():
        raise ValueError("Observed value must be a nonempty real Tensor")


def _output_specification(name: str, specification: Mapping[str, Any]) -> dict:
    """Copy and validate a hook selector before distributed training starts."""
    if not isinstance(specification, Mapping):
        raise ValueError(f"Output observation {name!r} must be a mapping")
    allowed = {"module", "selector", "broadcast_batch"}
    if specification.keys() - allowed or "module" not in specification:
        raise ValueError(
            "Output observation requires module and optional selector/broadcast_batch"
        )
    module = specification["module"]
    if not isinstance(module, str) or not module:
        raise ValueError("Output module must be a nonempty module path")
    selector = specification.get("selector", ())
    if not isinstance(selector, (list, tuple)) or any(
        type(part) not in (int, str) for part in selector
    ):
        raise ValueError("Output selector must contain integer indices or keys")
    broadcast = specification.get("broadcast_batch", False)
    if type(broadcast) is not bool:
        raise ValueError("broadcast_batch must be a boolean")
    return {"module": module, "selector": tuple(selector), "broadcast_batch": broadcast}


def _tensor_moments(value: torch.Tensor) -> torch.Tensor:
    """Pack detached FP64 [count, sum, squared sum, min, max] on the input device."""
    _validate_tensor(value)
    value = value.detach().to(torch.float64)
    return torch.stack(
        (
            value.new_tensor(value.numel()),
            value.sum(),
            value.square().sum(),
            value.amin(),
            value.amax(),
        )
    )


def _moment_statistics(moment: torch.Tensor) -> dict[str, float]:
    """Convert globally reduced moments to population statistics on the logging rank."""
    count, total, squared_total, minimum, maximum = moment.cpu().tolist()
    if count == 0:
        return {}
    mean = total / count
    # Clip round-off below zero while preserving NaN diagnostics.
    variance = max(squared_total / count - mean**2, 0.0)
    return {
        "mean": mean,
        "std": variance**0.5,
        "min": minimum,
        "max": maximum,
        "count": count,
    }


def tensor_statistics(value: torch.Tensor) -> dict[str, float]:
    """Summarize a nonempty real Tensor without retaining its computation graph."""
    _validate_tensor(value)
    value = value.detach().float()
    packed = (
        torch.stack((value.mean(), value.amin(), value.amax(), value.std(correction=0)))
        .cpu()
        .tolist()
    )
    return dict(zip(("mean", "min", "max", "std"), packed))


class TensorMonitor(Callback):
    """Log parameters, unscaled gradients and selected forward-output statistics.

    Paths are relative to the LightningModule, e.g. ``model.aggregator.p``.
    ``parameters`` maps metric names to exact parameter paths. ``outputs`` maps
    names to {module: path, selector: [index_or_key, ...]}; [1, "p"] selects
    aggregator auxiliary p. Arbitrary feature tensors use the same selector.

    Log initial parameters and then every_n_steps optimizer updates. Parameters
    are read after the update; gradients before clipping/step (after AMP unscale).
    Outputs aggregate every training forward across ranks and microbatches since
    the previous emission. Epoch end flushes the remaining partial interval.
    Mean/std are element-weighted population statistics; min/max are true global
    extrema. Groups are pooled as sample-group values. Set broadcast_batch=True
    for a shared [1,...] output to count it once per input sample. Validation and
    backward recomputation are excluded. Partial intervals restart on resume.
    No extra forward pass, tensor mutation, or graph retention is performed.
    Native loggers receive scalars; CSV and SwanLab need no special adapters.
    """

    def __init__(
        self,
        *,
        parameters: Mapping[str, str] | None = None,
        outputs: Mapping[str, Mapping[str, Any]] | None = None,
        every_n_steps: int = 50,
        gradients: bool = True,
        key: str = "observe",
    ) -> None:
        if type(every_n_steps) is not int or every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        self.parameters = dict(parameters or {})
        self.outputs = {
            name: _output_specification(name, specification)
            for name, specification in (outputs or {}).items()
        }
        if not self.parameters and not self.outputs:
            raise ValueError("TensorMonitor requires parameters or outputs")
        self.every_n_steps = every_n_steps
        self.gradients = gradients
        self.key = key
        self._handles: list[RemovableHandle] = []
        self._collect_outputs = False
        self._pending: dict[str, float] = {}
        self._parameters: dict[str, nn.Parameter] = {}
        self._moments: dict[str, torch.Tensor] = {}

    def _record(self, name: str, value: torch.Tensor) -> None:
        self._pending.update(
            {
                f"{self.key}/{name}/{stat}": number
                for stat, number in tensor_statistics(value).items()
            }
        )

    def _hook(self, name: str, selector: Sequence[int | str]) -> Callable:
        def observe(module, inputs, output):
            if self._collect_outputs:
                for part in selector:
                    output = output[part]
                _validate_tensor(output)
                if self.outputs[name]["broadcast_batch"]:
                    if output.ndim == 0 or output.shape[0] != 1:
                        raise ValueError(
                            "broadcast_batch requires output shape [1,...]"
                        )
                    if (
                        not inputs
                        or not isinstance(inputs[0], torch.Tensor)
                        or inputs[0].ndim == 0
                    ):
                        raise ValueError(
                            "broadcast_batch requires a batched first positional "
                            "Tensor input"
                        )
                    output = output.expand(inputs[0].shape[0], *output.shape[1:])
                self._accumulate(name, output)

        return observe

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._remove_hooks()
        # Resolve on every rank so a typo fails consistently before training.
        self._parameters = {
            name: pl_module.get_parameter(path)
            for name, path in self.parameters.items()
        }
        modules = {
            name: pl_module.get_submodule(spec["module"])
            for name, spec in self.outputs.items()
        }
        for name, module in modules.items():
            self._handles.append(
                module.register_forward_hook(
                    self._hook(name, self.outputs[name].get("selector", []))
                )
            )

    def _accumulate(self, name: str, value: torch.Tensor) -> None:
        moment = _tensor_moments(value)
        if name not in self._moments:
            self._moments[name] = moment
        else:
            previous = self._moments[name]
            previous[:3].add_(moment[:3])
            previous[3] = torch.minimum(previous[3], moment[3])
            previous[4] = torch.maximum(previous[4], moment[4])

    def _flush_outputs(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Every rank visits every configured output, including unexecuted branches.
        for name in self.outputs:
            moment = self._moments.get(name)
            if moment is None:
                moment = torch.tensor(
                    [0, 0, 0, float("inf"), -float("inf")],
                    device=pl_module.device,
                    dtype=torch.float64,
                )
            totals = trainer.strategy.reduce(moment[:3].clone(), reduce_op=ReduceOp.SUM)
            minimum = trainer.strategy.reduce(moment[3].clone(), reduce_op=ReduceOp.MIN)
            maximum = trainer.strategy.reduce(moment[4].clone(), reduce_op=ReduceOp.MAX)
            if trainer.is_global_zero:
                reduced = torch.cat((totals, minimum.reshape(1), maximum.reshape(1)))
                statistics = _moment_statistics(reduced)
                self._pending.update(
                    {
                        f"{self.key}/output/{name}/{stat}": value
                        for stat, value in statistics.items()
                    }
                )
        self._moments.clear()

    def _emit(self, trainer: Trainer) -> None:
        if self._pending:
            for logger in trainer.loggers:
                logger.log_metrics(dict(self._pending), step=trainer.global_step)
        self._pending.clear()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.is_global_zero:
            for name, parameter in self._parameters.items():
                self._record(f"parameter/{name}", parameter)
            self._emit(trainer)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        self._pending.clear()
        self._start_step = trainer.global_step
        self._collect_outputs = bool(trainer.loggers)
        self._emit_after_update = (self._start_step + 1) % self.every_n_steps == 0

    def on_before_backward(
        self, trainer: Trainer, pl_module: LightningModule, loss: torch.Tensor
    ) -> None:
        # Activation-checkpoint recomputation must not count samples twice.
        self._collect_outputs = False

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if trainer.is_global_zero and self._emit_after_update and self.gradients:
            for name, parameter in self._parameters.items():
                # Missing gradients are distinct from actual zero gradients.
                self._pending[f"{self.key}/gradient/{name}/present"] = float(
                    parameter.grad is not None
                )
                if parameter.grad is not None:
                    self._record(f"gradient/{name}", parameter.grad)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._collect_outputs = False
        if self._emit_after_update and trainer.global_step > self._start_step:
            self._flush_outputs(trainer, pl_module)
            if trainer.is_global_zero:
                for name, parameter in self._parameters.items():
                    self._record(f"parameter/{name}", parameter)
                self._emit(trainer)
        self._pending.clear()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._flush_outputs(trainer, pl_module)
        if trainer.is_global_zero:
            self._emit(trainer)

    def _remove_hooks(self) -> None:
        self._collect_outputs = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending.clear()
        self._parameters.clear()
        self._moments.clear()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._remove_hooks()

    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        self._remove_hooks()

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        self._remove_hooks()
