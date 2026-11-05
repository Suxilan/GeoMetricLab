"""Check the installed library and optional FAISS after normal environment activation."""

import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys

import torch
import gmlab


def main() -> None:
    distribution = metadata.distribution("geometriclab")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    result = {
        "python": sys.executable,
        "gmlab": gmlab.__file__,
        "version": distribution.version,
        "editable": direct.get("dir_info", {}).get("editable", False),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if importlib.util.find_spec("faiss") is None:
        result["faiss"] = "not installed; Torch retrieval remains available"
    else:
        # Torch deliberately loads first to catch C++ runtime order conflicts.
        import faiss
        import numpy as np

        values = np.random.default_rng(0).normal(size=(64, 16)).astype("float32")
        index = faiss.IndexFlatL2(16)
        index.add(values)
        distances, indices = index.search(values[:4], 3)
        np.testing.assert_array_equal(indices[:, 0], np.arange(4))
        gpu_count = faiss.get_num_gpus()
        result["faiss"] = {
            "version": faiss.__version__,
            "cpu_search": "passed",
            "visible_gpus": gpu_count,
        }
        if gpu_count:
            resources = faiss.StandardGpuResources()
            resources.setTempMemory(16 * 1024 * 1024)
            gpu = faiss.index_cpu_to_gpu(resources, 0, index)
            gpu_distances, gpu_indices = gpu.search(values[:4], 3)
            np.testing.assert_array_equal(indices, gpu_indices)
            np.testing.assert_allclose(distances, gpu_distances, atol=1e-4)
            result["faiss"]["gpu_search"] = "passed"
    maps = Path("/proc/self/maps")
    if maps.is_file():
        result["cxx_runtime"] = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if "libstdc++" in line
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
