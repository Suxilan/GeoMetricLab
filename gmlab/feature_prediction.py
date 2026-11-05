"""Lightning prediction launching for the shared bounded extraction producer."""

import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset

from .utils.distributed import distributed_active, rank
from .utils.io import atomic_json
from .utils.precision import trainer_inference_options


class _ExtractionJob(IterableDataset):
    """Give every prediction rank exactly one job, without sampler padding."""

    def __iter__(self):
        yield 0


class FeaturePrediction(pl.LightningModule):
    """Use native Lightning process/device setup for one bounded extraction job.

    The prediction result is a small record, never a list of feature batches.
    Tensor inference, chunk transport, staging and publication all use the same
    producer as direct Python/torchrun calls. Lightning owns process launch and
    teardown; no optimizer, gradient synchronization or custom Trainer is added.
    """

    def __init__(self, model, loader, options, record_path, producer):
        super().__init__()
        self.model = model
        self.producer = producer
        self.input_loader = loader
        self.options = options
        self.record_path = record_path
        self.result = None

    def __getstate__(self):
        state = super().__getstate__()
        # PyTorch 2.10's multiprocessing reducer can lose the temporary storage
        # returned by Generator.get_state(), even for a plain native Generator.
        # Serialize the loader with Python's value-based protocol, preserving
        # sampler/generator aliases and RNG state without changing global reducers.
        # The model still uses Lightning/PyTorch's native tensor transport.
        state["input_loader"] = pickle.dumps(
            self.input_loader, protocol=pickle.HIGHEST_PROTOCOL
        )
        return state

    def __setstate__(self, state):
        state["input_loader"] = pickle.loads(state["input_loader"])
        super().__setstate__(state)

    def predict_step(self, batch, batch_idx):
        self.result = self.producer(
            model=self.model,
            loader=self.input_loader,
            device=self.device,
            **self.options,
        )
        if rank() == 0:
            atomic_json(self.record_path, self.result)
        return None


def predict_extraction(*, producer, model, loader, trainer, **options):
    """Launch shared extraction using an existing native Lightning Trainer.

    Every rank receives a single job token, including ranks with zero image
    rows. Internal input sharding is exact and does not pad duplicate samples.
    Inherit native Trainer precision; reject a conflicting producer request.
    Spawn workers return the small JSON record through a private temporary file;
    feature tensors are never retained by Lightning or returned to the parent.
    Spawn serializes the input loader by value, including Tensor-backed dataset
    payloads. Use subprocess DDP for datasets with large in-memory Tensor state.
    """
    options.update(
        trainer_inference_options(
            trainer,
            precision=options.get("precision"),
            matmul_tf32=options.get("matmul_tf32"),
            cudnn_tf32=options.get("cudnn_tf32"),
        )
    )
    # Preserve direct single-device use, including cache hits without starting
    # a prediction loop. Multi-device runs enter after native process launch.
    if trainer.num_devices == 1 or distributed_active():
        return producer(
            model=model, loader=loader, device=trainer.strategy.root_device, **options
        )
    with TemporaryDirectory(prefix="gmlab-predict-") as temporary:
        record_path = Path(temporary) / "record.json"
        module = FeaturePrediction(model, loader, options, record_path, producer)
        trainer.predict(
            module,
            DataLoader(_ExtractionJob(), batch_size=1, num_workers=0),
            return_predictions=False,
        )
        # In subprocess DDP each launched process keeps its local module. In
        # spawn/fork the launcher parent instead reads rank zero's tiny record.
        if module.result is not None:
            return module.result
        return json.loads(record_path.read_text())
