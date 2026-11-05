"""Explicit aligned stream mixtures for shared affine fitting."""

from gmlab.datasets.utils.feature_io import feature_collection
from gmlab.datasets.utils.feature_description import validate_feature_space


def aligned_stream_collection(selections, *, streams, cache_root):
    """Combine declared outputs with equal row weights and verified row identity.

    Each consecutive group names streams in the configured order. Model state,
    preprocessing, shape and output semantics must agree except for the explicit
    stream/branch names. This is a consumer policy; ordinary cache compatibility
    remains strict. No modality or fixed number of branches is assumed.
    """
    if (
        not isinstance(streams, (list, tuple))
        or len(streams) < 2
        or any(not isinstance(name, str) or not name for name in streams)
        or len(set(streams)) != len(streams)
        or not selections
        or len(selections) % len(streams)
    ):
        raise ValueError("Declare distinct streams and complete aligned groups")
    records = {name: [] for name in streams}
    for index, item in enumerate(selections):
        name = streams[index % len(streams)]
        if item.get("stream") != name or "record" not in item:
            raise ValueError(f"Feature input {index} requires record and stream={name}")
        records[name].append(item["record"])
    groups = {
        name: feature_collection(
            records=records[name],
            stream=name,
            cache_root=cache_root,
            expected_output={
                "layout": "global",
                "normalization": "none",
                "whitening": False,
            },
        )
        for name in streams
    }
    reference = groups[streams[0]]
    for name in streams[1:]:
        calculation = groups[name]["calculation"]
        output = dict(calculation["output"])
        for key in ("stream", "branch"):
            if key in reference["calculation"]["output"]:
                output[key] = reference["calculation"]["output"][key]
        validate_feature_space(
            {**calculation, "output": output}, reference["calculation"]
        )
    inputs = []
    for aligned in zip(*(groups[name]["inputs"] for name in streams)):
        first = aligned[0]
        identity = first["manifest"]["identity"]
        for item in aligned[1:]:
            other = item["manifest"]["identity"]
            digest = identity["calculation"].get("samples_sha256")
            if (
                not digest
                or digest != other["calculation"].get("samples_sha256")
                or identity["source"] != other["source"]
                or first["shape"] != item["shape"]
                or first["image_names"] != item["image_names"]
            ):
                raise ValueError("Aligned stream image selection/order differs")
        inputs.extend(aligned)
    keys = [item["manifest"]["artifact_key"] for item in inputs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate feature artifact in aligned stream selection")
    return {
        "inputs": inputs,
        "artifact_keys": keys,
        "shape": [sum(item["shape"][0] for item in inputs), reference["shape"][1]],
        "cache_root": reference["cache_root"],
        "combination": "aligned_streams",
        "calculation": {
            "combination": "aligned_streams",
            "sampling": "all_rows",
            "weighting": "all_selected_rows_equal",
            "spaces": {name: groups[name]["calculation"] for name in streams},
        },
    }
