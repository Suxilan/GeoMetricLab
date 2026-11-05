"""Load local DINOv3 backbones without importing unrelated task dependencies."""

from importlib import import_module
from pathlib import Path
import sys


def load_dinov3(repository, name, **options):
    """Call the upstream backbone factory from an explicit repository.

    DINOv3's hubconf also imports segmentation code. Metric encoders only need
    its backbone factories. Reject a cached package from a different checkout
    rather than silently resolving weights and code against different sources.
    """
    root = Path(repository).expanduser().resolve()
    if not (root / "dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError(f"DINOv3 backbone repository not found: {root}")
    sys.path.insert(0, str(root))
    try:
        module = import_module("dinov3.hub.backbones")
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("DINOv3 was already imported from another repository")
        return getattr(module, name)(**options)
    finally:
        sys.path.remove(str(root))
