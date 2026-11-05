"""Explicit construction at configuration boundaries, without plugin scanning."""

from collections.abc import Mapping
from importlib import import_module
from typing import Any, Callable

ADAPTERS = {
    "grouped_inference": "gmlab.training.adapters.grouped_inference_batch",
    "grouped_named": "gmlab.training.adapters.grouped_named_batch",
    "stacked_id": "gmlab.training.adapters.stacked_id_batch",
    "named": "gmlab.training.adapters.named_batch",
    "named_outputs": "gmlab.training.adapters.named_embeddings",
    "modality_id": "gmlab.training.adapters.modality_id_batch",
    "cross_modal_id": "gmlab.training.adapters.cross_modal_id_batch",
    "modality_image": "gmlab.training.adapters.modality_image_batch",
    "id": "gmlab.training.adapters.id_batch",
    "multiview_id": "gmlab.training.adapters.multiview_id_batch",
    "relation": "gmlab.training.adapters.relation_batch",
    "image": "gmlab.training.adapters.image_batch",
    "tensor": "gmlab.training.adapters.tensor_embeddings",
    "descriptor": "gmlab.training.adapters.descriptor_embeddings",
}


# Import paths keep optional model/data dependencies unloaded until construction.
_COMPONENTS = {
    "GeoPipeline": "gmlab.pipeline.GeoPipeline",
    "MultiDatasetPipeline": "gmlab.pipeline.MultiDatasetPipeline",
    "MultimodalPipeline": "gmlab.pipeline.MultimodalPipeline",
    "ProjectionHead": "gmlab.models.projection.ProjectionHead",
    "RetrievalHead": "gmlab.models.projection.RetrievalHead",
    "MoCoProjection": "gmlab.models.projection.MoCoProjection",
    "Salad": "gmlab.models.aggregator.salad.Salad",
    "DinoAggregation": "gmlab.models.aggregator.dino.DinoAggregation",
    "Avg": "gmlab.models.aggregator.avg.Avg",
    "BoQ": "gmlab.models.aggregator.boq.BoQ",
    "CFP": "gmlab.models.aggregator.cfp.CFP",
    "CGeM": "gmlab.models.aggregator.cgem.CGeM",
    "CLS": "gmlab.models.aggregator.cls.CLS",
    "ConvNeXt": "gmlab.models.backbone.convnext.ConvNeXt",
    "CosPlace": "gmlab.models.aggregator.cosplace.CosPlace",
    "CricaVPR": "gmlab.models.aggregator.cricavpr.CricaVPR",
    "DGeM": "gmlab.models.aggregator.dgem.DGeM",
    "DepthAnythingV2Backbone": "gmlab.models.backbone.depth_anything_v2.DepthAnythingV2Backbone",
    "DinoV2": "gmlab.models.backbone.dinov2.DinoV2",
    "EigenPlace": "gmlab.models.aggregator.eigenplace.EigenPlace",
    "GhostVLAD": "gmlab.models.aggregator.ghostvlad.GhostVLAD",
    "NetVLAD": "gmlab.models.aggregator.netvlad.NetVLAD",
    "PEFTDinoV3": "gmlab.models.backbone.peft_dinov3.PEFTDinoV3",
    "RegVPR": "gmlab.models.aggregator.regvpr.RegVPR",
    "SoftPCFP": "gmlab.models.aggregator.softp.SoftPCFP",
    "SuperVLAD": "gmlab.models.aggregator.supervlad.SuperVLAD",
    "SwinTransformer": "gmlab.models.backbone.swin_transformer.SwinTransformer",
    "Linear": "torch.nn.Linear",
    "stack_samples": "gmlab.datasets.utils.samples.stack_samples",
    "group_samples": "gmlab.datasets.utils.samples.group_samples",
    "group_cross_modal_samples": "gmlab.datasets.utils.samples.group_cross_modal_samples",
    "modality_collate": "gmlab.datasets.utils.multimodal.modality_collate",
    "AnyLoc": "gmlab.models.anyloc.build_anyloc",
    "ModalityEncoder": "gmlab.models.multimodal.build_modality_encoder",
    "NamedBranches": "gmlab.models.branches.build_named_branches",
    "Affine": "gmlab.models.assembly.build_affine",
    "pooled_descriptor": "gmlab.models.branches.pooled_descriptor",
    "descriptor_features": "gmlab.models.multimodal.descriptor_features",
    "GeoEncoder": "gmlab.models.geoencoder.build_geoencoder",
    "select_feature_streams": "gmlab.models.outputs.select_feature_streams",
    "spatial_features": "gmlab.models.geoencoder.spatial_features",
    "dino_patch_features": "gmlab.models.geoencoder.dino_patch_features",
    "patch_tokens": "gmlab.models.geoencoder.patch_tokens",
    "DinoV3": "gmlab.models.backbone.dinov3.DinoV3",
    "SuperGeM": "gmlab.models.aggregator.supergem.SuperGeM",
    "ResNet": "gmlab.models.backbone.resnet.ResNet",
    "PEFTDinoV2": "gmlab.models.backbone.peft_dinov2.PEFTDinoV2",
    "GeM": "gmlab.models.aggregator.gem.GeM",
    "SCPP": "gmlab.models.aggregator.scpp.SCPP",
    "MetricObjective": "gmlab.training.metric.MetricObjective",
    "PMLObjective": "gmlab.losses.objectives.PMLObjective",
    "BoundObjective": "gmlab.losses.bindings.BoundObjective",
    "GeoMetricLoss": "gmlab.losses.geo_metric_loss.GeoMetricLoss",
    "OverlapRankingObjective": "gmlab.losses.pml.overlap_ranking.OverlapRankingObjective",
    "SubgraphDatasetCollection": "gmlab.datasets.utils.supscene.collection.SubgraphDatasetCollection",
    "ClassDatasetCollection": "gmlab.datasets.utils.classification.ClassDatasetCollection",
    "SubgraphSampler": "gmlab.datasets.utils.supscene.subgraph_sampler.SubgraphSampler",
    "SequentialSampler": "torch.utils.data.SequentialSampler",
    "RandomSampler": "torch.utils.data.RandomSampler",
    "BatchSampler": "torch.utils.data.BatchSampler",
    "SceneEpochSampler": "gmlab.datasets.utils.sampling.GeoEpochSampler",
    "GeoEpochSampler": "gmlab.datasets.utils.sampling.GeoEpochSampler",
    "PadSubgraphs": "gmlab.datasets.utils.supscene.collate.make_pad_collate",
    "Adam": "torch.optim.Adam",
    "AdamW": "torch.optim.AdamW",
    "SGD": "torch.optim.SGD",
    "StepLR": "torch.optim.lr_scheduler.StepLR",
    "WarmupCosine": "gmlab.training.optimization.warmup_cosine",
    "EarlyStopping": "pytorch_lightning.callbacks.EarlyStopping",
    "ModalityQuotaSampler": "gmlab.datasets.utils.modality_sampling.ModalityQuotaSampler",
    "CSVLogger": "pytorch_lightning.loggers.CSVLogger",
}


def register_component(name: str, constructor: Callable | str) -> None:
    """Register an explicit extension, rejecting accidental name replacement."""
    if name in _COMPONENTS:
        raise ValueError(f"Component already registered: {name}")
    if not callable(constructor) and not isinstance(constructor, str):
        raise TypeError("Component must be a callable or an import path")
    _COMPONENTS[name] = constructor


def resolve_component(name: str) -> Callable:
    """Resolve a registered name or a fully qualified external import path."""
    if not isinstance(name, str):
        raise TypeError("Component type must be a registered name or an import path")
    target = _COMPONENTS.get(name, name)
    if callable(target):
        return target
    if "." not in target:
        raise ValueError(
            f"Unknown component {name!r}; registered: {sorted(_COMPONENTS)}"
        )
    module, attribute = target.rsplit(".", 1)
    constructor = getattr(import_module(module), attribute)
    if not callable(constructor):
        raise TypeError(f"Component is not callable: {target}")
    return constructor


def component_definition(component: Mapping) -> tuple[Callable, dict[str, Any]]:
    """Resolve a constructor and arguments without constructing its object.

    The type/arguments form is the YAML interface. The explicit
    module/class/params form is decoded here for existing Python consumers;
    no caller needs to interpret either form again.
    """
    if "type" in component:
        if {"module", "class", "params"} & component.keys():
            raise ValueError("Do not mix type/arguments with module/class/params")
        constructor = resolve_component(component["type"])
        arguments = {key: value for key, value in component.items() if key != "type"}
    else:
        unknown = component.keys() - {"module", "class", "params"}
        if unknown or not {"module", "class"} <= component.keys():
            raise ValueError("Expected type and arguments")
        constructor = resolve_component(component["module"] + "." + component["class"])
        arguments = dict(component.get("params", {}))
    return constructor, arguments


def build_component(component: Any, **injected: Any) -> Any:
    """Return an instance or construct an explicitly described component.

    Instances are returned unchanged. Injected arguments (e.g. optimizer params)
    are only accepted for specifications and cannot override configured values.
    YAML uses ``{type: RegisteredName, argument: value}``. External Python paths
    are also accepted as type names. The earlier module/class/params form remains
    an explicit low-level construction option. Nested mappings are not guessed
    as components; the owning constructor decides which arguments are components.
    """
    if not isinstance(component, Mapping):
        if injected:
            raise TypeError("Injected arguments require a component specification")
        return component
    constructor, arguments = component_definition(component)
    if arguments.keys() & injected.keys():
        raise ValueError("Configured parameters conflict with injected arguments")
    return constructor(**arguments, **injected)


def resolve_adapter(name: str) -> Callable:
    """Resolve a built-in adapter name or an external callable path."""
    return resolve_component(ADAPTERS.get(name, name))
