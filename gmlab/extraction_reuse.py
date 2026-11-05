"""Copy verified immutable feature rows into caller-owned private staging.

The extraction runtime owns transactions and source certification. This helper
only selects compatible published rows and copies them at stable target indices.
"""

import json
from pathlib import Path

import numpy as np

from .artifacts import identity_key
from .datasets.utils.feature_io import GlobalFeatureWriter


def _reuse_rows(
    store, identity, samples, writer: GlobalFeatureWriter, *, chunk_rows=256
):
    """Copy matching per-image results from compatible immutable feature sets.

    Dataset names and row order describe a selection, not a numerical feature
    space. Content plus crop/view identifies a reusable input row. No per-image
    filesystem hierarchy or second cache database is introduced.
    """
    from .datasets.utils.feature_io import iter_global_features
    from .datasets.utils.feature_description import recorded_compute

    calculation = identity["calculation"]
    numerical = {
        k: v
        for k, v in calculation.items()
        if k not in {"samples_sha256", "num_samples"}
    }

    numerical["compute"] = recorded_compute(calculation)
    if numerical["compute"] is None:
        return 0, []
    numerical_key = identity_key(numerical)

    def row_key(sample):
        return identity_key(
            {k: sample.get(k) for k in ("content_sha256", "region", "view")}
        )

    needed = {}
    written = writer.file["written"][:]
    for index, sample in enumerate(samples):
        if not written[index]:
            needed.setdefault(row_key(sample), []).append(index)
    reused, sources = 0, []
    for saved in store.inventory():
        if not needed:
            break
        candidate = saved.get("identity", {})
        if candidate.get("kind") != identity["kind"]:
            continue
        other = {
            k: v
            for k, v in candidate.get("calculation", {}).items()
            if k not in {"samples_sha256", "num_samples"}
        }
        other["compute"] = recorded_compute(
            candidate.get("calculation", {}), manifest=saved
        )
        # Compare through the same canonical form as artifact identity. JSON
        # reloads tuples (e.g. constructor defaults) as lists, not new math.
        if identity_key(other) != numerical_key:
            continue
        directory = Path(saved["path"])
        rows = [
            json.loads(line)
            for line in (directory / "samples.jsonl").read_text().splitlines()
        ]
        matches = {
            i: row_key(row) for i, row in enumerate(rows) if row_key(row) in needed
        }
        if not matches:
            continue
        manifest = store.verify(directory)
        payload = next(
            p
            for p in manifest["payloads"]
            if p.get("format") == calculation["storage"]["format"]
        )
        copied = 0
        for start, values in iter_global_features(
            directory / payload["path"], chunk_rows=chunk_rows
        ):
            for offset, value in enumerate(values):
                key = matches.get(start + offset)
                targets = needed.pop(key, [])
                # One cached row may satisfy many intentional repeated inputs.
                # Bound that fan-out too; do not materialize N repeated maps.
                for target_start in range(0, len(targets), chunk_rows):
                    indices = np.asarray(
                        targets[target_start : target_start + chunk_rows]
                    )
                    writer.write(
                        indices, np.broadcast_to(value, (len(indices), *value.shape))
                    )
                    copied += len(indices)
        if copied:
            sources.append(manifest["artifact_key"])
            reused += copied
    return reused, sources
