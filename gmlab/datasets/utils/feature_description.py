"""Readable views of declared feature calculations; no inference or mutation."""


def describe_feature(manifest):
    """Summarize manifest preprocessing without inventing a parallel identity.

    This is a view of declarations, not an integrity check. Full ordered
    preprocessing remains included; absent historical metadata is unknown,
    never interpreted as original resolution.
    """
    identity = manifest.get("identity", {})
    calculation = identity.get("calculation", {})
    preprocessing = calculation.get("preprocessing")
    operations = (preprocessing or {}).get("operations")
    if operations is None:
        resize = "unknown"
    else:
        sizes = [
            op.get("size") for op in operations if op["type"].split(".")[-1] == "Resize"
        ]
        resize = sizes if sizes else "none"
    return {
        "data_name": identity.get("source"),
        "method": identity.get("selection"),
        "artifact_key": manifest.get("artifact_key"),
        "resize": resize,
        "preprocessing": preprocessing,
        "output": calculation.get("output"),
        "payloads": [
            {k: payload[k] for k in ("path", "shape", "dtype") if k in payload}
            for payload in manifest.get("payloads", [])
        ],
    }


FEATURE_SPACE_FIELDS = (
    "model",
    "model_modules",
    "state_sha256",
    "adapter",
    "input_adapter",
    "preprocessing",
    "output",
    "storage",
    "algorithm",
    "parents",
    "realization",
    "seed",
    "compute",
)


def feature_space(calculation, *, manifest=None):
    """Return the numerical space, excluding dataset selection and runtime facts.

    Seed and realization remain strict declarations. Missing historical values
    stay unknown; they are not silently upgraded to current defaults.
    """
    result = {field: calculation.get(field) for field in FEATURE_SPACE_FIELDS}
    result["compute"] = recorded_compute(calculation, manifest=manifest)
    return result


def validate_feature_space(actual, expected):
    """Reject differing calculation fields, including nested output semantics."""
    from ...artifacts import canonical_json

    def differences(left, right, prefix):
        if isinstance(left, dict) and isinstance(right, dict):
            result = []
            for key in sorted(left.keys() | right.keys()):
                result.extend(
                    differences(left.get(key), right.get(key), f"{prefix}.{key}")
                )
            return result
        return [] if canonical_json(left) == canonical_json(right) else [prefix]

    if recorded_compute(actual) is None or recorded_compute(expected) is None:
        raise ValueError(
            "Cannot combine different feature spaces: unknown calculation.compute precision"
        )
    fields = differences(feature_space(actual), feature_space(expected), "calculation")
    if fields:
        raise ValueError(
            "Cannot combine different feature spaces: " + ", ".join(fields)
        )


def validate_feature_output(output, row_shape):
    """Check declared per-image output layout/shape against the actual payload.

    shape excludes the sample axis: [D] for global and [C,H,W] for spatial.
    Historical declarations without shape remain valid; an explicit declaration
    must be positive integer dimensions and match the actual H5 schema.
    """
    if "shape" in output:
        declared = output["shape"]
        if (
            not isinstance(declared, (list, tuple))
            or not declared
            or any(type(value) is not int or value < 1 for value in declared)
            or list(declared) != list(row_shape)
        ):
            raise ValueError(
                "Actual feature shape differs from calculation.output.shape"
            )
    rank = {"global": 1, "spatial": 3}.get(output.get("layout"))
    if rank is not None and len(row_shape) != rank:
        raise ValueError("Actual feature layout differs from calculation.output.layout")
    dimension = output.get("dimension")
    if dimension is not None and (
        type(dimension) is not int or dimension < 1 or list(row_shape) != [dimension]
    ):
        raise ValueError("Actual output dimension differs from declared output space")


def recorded_compute(calculation, *, manifest=None):
    """Read explicit compute policy or documented historical FP32 execution.

    Missing metadata stays unknown. Only the former forced-FP32 producer and
    its actual numeric execution record justify a historical interpretation;
    file/storage dtype alone is never evidence of calculation precision.
    """
    if "compute" in calculation:
        policy = calculation["compute"]
        fields = {"precision", "matmul_tf32", "cudnn_tf32", "float32_matmul_precision"}
        if not isinstance(policy, dict) or set(policy) != fields:
            return None
        if policy["precision"] not in ("32-true", "bf16-mixed", "16-mixed"):
            return None
        if any(type(policy[key]) is not bool for key in ("matmul_tf32", "cudnn_tf32")):
            return None
        if policy["float32_matmul_precision"] != (
            "high" if policy["matmul_tf32"] else "highest"
        ):
            return None
        return dict(policy)
    if (
        manifest is None
        or manifest.get("provenance_status") != "verified"
        or calculation.get("algorithm") != "global_fp32_model_adapter_v1"
    ):
        return None
    numeric = manifest.get("metadata", {}).get("execution", {}).get("numeric", {})
    if numeric.get("precision") != "float32":
        return None
    if numeric.get("backend") == "cpu":
        if numeric.get("float32_matmul_precision") != "highest":
            return None
        flags = False, False
    elif numeric.get("backend") == "cuda":
        flags = numeric.get("matmul_tf32"), numeric.get("cudnn_tf32")
        if any(type(value) is not bool for value in flags):
            return None
    else:
        return None
    mode = "high" if flags[0] else "highest"
    if numeric.get("float32_matmul_precision") != mode:
        return None
    return {
        "precision": "32-true",
        "matmul_tf32": flags[0],
        "cudnn_tf32": flags[1],
        "float32_matmul_precision": mode,
    }
