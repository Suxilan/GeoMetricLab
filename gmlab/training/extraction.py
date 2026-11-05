"""Map evaluation recipe choices to the public feature extraction API."""

from ..artifacts import ArtifactStore
from ..extraction import predict_extraction


def extract_artifact(config, loader, module, trainer, *, provenance=None):
    """Map explicit recipe choices onto the common multi-device producer."""
    options = config["artifacts"]
    result = predict_extraction(
        model=module.model,
        model_config=config["model"],
        loader=loader,
        trainer=trainer,
        embedding_adapter=module.inference_embedding_adapter,
        batch_adapter=module.inference_batch_adapter,
        input_root=options["input_root"],
        data_name=options.get("data_name", options.get("source")),
        method_name=options.get("method_name", type(module.model).__name__.lower()),
        store=ArtifactStore(options.get("root")),
        storage_dtype=options.get("dtype", "float32"),
        output_space=options.get("output_space"),
        stream_specs=options.get("streams"),
        parents=options.get("parents", ()),
        seed=options.get("seed", 0),
        realization=options.get("realization", "deterministic_seed_v1"),
        provenance=provenance,
        chunk_rows=options.get("chunk_rows", 64),
        **module.inference_options,
    )
    return result if options.get("streams") is not None else result["streams"]["global"]
