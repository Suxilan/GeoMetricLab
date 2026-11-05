"""Deterministic, GT-free image inputs for feature extraction."""

from collections.abc import Mapping

from PIL import Image
from pathlib import Path

from .geo_dataset import GeoDataset
from .identities import integer_tensor


def load_rgb_image(row: Mapping) -> Image.Image:
    """Decode one path input as RGB, then apply its optional pixel crop.

    The returned image owns its pixels; the source file is closed before return.
    """
    with Image.open(row["inputs"]) as image:
        image = image.convert("RGB")
        region = row.get("region")
        return image if region is None else image.crop(region)


class ImageList(GeoDataset):
    """Return ``images [C,H,W], index (int)`` in immutable path-list order.

    ``transform`` converts RGB PIL images to tensors. It may be a mapping from
    explicit modalities to callables; use modality_collate for differing shapes.
    Optional modalities align with paths; unspecified rows use modality "image".
    The public modalities attribute retains None when no assignments were supplied.
    sample_id is ("image", resolved physical path); repeats share identity while
    indices identify entries in this list, including intentional repeated paths.
    Crops remain input annotations, independent of physical image identity.
    Retrieval roles and GT are separate metadata owned by the caller.
    """

    def __init__(self, paths, transform, regions=None, *, modalities=None):
        if isinstance(paths, (str, bytes, Path)):
            raise TypeError("paths must be an iterable of image paths, not one path")
        if isinstance(modalities, (str, bytes)):
            raise TypeError("modalities must be an iterable of modality names")
        self.paths = tuple(str(path) for path in paths)
        self.transform = transform
        self.modalities = None if modalities is None else tuple(modalities)
        if self.modalities is not None and len(self.modalities) != len(self.paths):
            raise ValueError("Modalities must align with image rows")
        if isinstance(transform, Mapping):
            if self.modalities is None:
                raise ValueError("Per-modality transforms require explicit modalities")
            missing = set(self.modalities) - transform.keys()
            if missing:
                raise ValueError(
                    f"Missing transforms for modalities: {sorted(missing)}"
                )
        self.regions = (
            None
            if regions is None
            else tuple(None if box is None else tuple(box) for box in regions)
        )
        if self.regions is not None and len(self.regions) != len(self.paths):
            raise ValueError("Regions must align with image rows")
        declared = tuple(dict.fromkeys(self.modalities or ("image",)))
        super().__init__(modalities=declared, transform=transform)

    def __len__(self):
        return len(self.paths)

    def _validate_feature_reader(self):
        """Require a new explicit declaration when sample computation changes."""
        for name, implementation in (
            ("__getitem__", ImageList.__getitem__),
            ("__getitems__", getattr(ImageList, "__getitems__", None)),
            ("load_input", ImageList.load_input),
            ("prepare_rows", GeoDataset.prepare_rows),
            ("_modality_transform", GeoDataset._modality_transform),
        ):
            operation = getattr(self, name, None)
            # PyTorch enables batched fetching only for a truthy __getitems__.
            if name == "__getitems__" and not operation:
                continue
            if getattr(operation, "__func__", operation) is not implementation:
                raise ValueError(
                    "Overridden image computation requires explicit feature_samples "
                    "and feature_preprocessing declarations"
                )

    @property
    def feature_preprocessing(self):
        """Describe the current decoder and transforms, without a stale snapshot."""
        from .feature_inputs import describe_preprocessing

        self._validate_feature_reader()
        return describe_preprocessing(self.transform)

    def feature_samples(self, *, source, input_root):
        """Describe stable path/crop/modality rows in this extraction's scope."""
        from .feature_inputs import image_feature_samples

        self._validate_feature_reader()
        return image_feature_samples(self, source=source, input_root=input_root)

    def load_input(self, row: Mapping) -> Image.Image:
        """Use the shared RGB decoder without changing manifest row order."""
        return load_rgb_image(row)

    def __getitem__(self, index):
        index_tensor = integer_tensor(index, name="Image index", minimum=0)
        if index_tensor.ndim:
            raise ValueError("Image index must be scalar")
        index = int(index_tensor)
        if index >= len(self):
            raise IndexError(index)
        selected = {
            "inputs": self.paths[index],
            "sample_id": ("image", str(Path(self.paths[index]).resolve())),
            "modality": (
                "image" if self.modalities is None else self.modalities[index]
            ),
        }
        if self.regions is not None:
            selected["region"] = self.regions[index]
        prepared = self.prepare_rows([selected])[0]
        return {
            "images": prepared["inputs"],
            "index": index,
            "sample_id": prepared["sample_id"],
            "modality": prepared["modality"],
        }


def find_images(roots):
    """Discover images deterministically, following aliases once without loops."""
    import os
    from pathlib import Path

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images, directories, files_seen = [], set(), set()
    for root in map(Path, roots):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for directory, children, files in os.walk(root, followlinks=True):
            resolved = Path(directory).resolve()
            if resolved in directories:
                children[:] = []
                continue
            directories.add(resolved)
            children.sort()
            for name in sorted(files):
                path = Path(directory) / name
                physical = path.resolve()
                if (
                    path.suffix.lower() in extensions
                    and path.is_file()
                    and physical not in files_seen
                ):
                    images.append(path)
                    files_seen.add(physical)
    return sorted(images)


def image_dataset_from_reader(root, transform, *, reader, reader_options=None):
    """Adapt an injected source reader to the common image/index contract.

    reader(root=..., **options) returns paths, input_root and optional regions,
    modalities and metadata. Paths are absolute or relative to input_root.
    Protocol decoding and query/gallery semantics belong to the source reader;
    this adapter only preserves its ordered image rows and explicit annotations.
    """
    from pathlib import Path
    from ...components import resolve_component

    if isinstance(reader, str):
        reader = resolve_component(reader)
    inputs = reader(root=root, **(reader_options or {}))
    if set(inputs) - {"paths", "input_root", "regions", "modalities", "metadata"}:
        raise ValueError("Unknown image-reader fields")
    input_root = Path(inputs["input_root"]).resolve()
    dataset = ImageList(
        [input_root / path for path in inputs["paths"]],
        transform,
        regions=inputs.get("regions"),
        modalities=inputs.get("modalities"),
    )
    dataset.input_root = input_root
    dataset.metadata = inputs.get("metadata", {})
    return dataset


def image_dataset(
    root, transform, *, image_list=None, name_files=(), unique=False, limit=None, seed=0
):
    """Construct GT-free images from an explicit list, manifests or a directory.

    Text image_list is relative to the launch directory; name_files are relative
    to root. Image names are relative to root. Repeated rows remain intentional
    unless unique=True. Selection is independent of model and downstream task.
    """
    from pathlib import Path
    import numpy as np

    root = Path(root).resolve()
    if image_list and name_files:
        raise ValueError("Choose image_list or name_files")
    if image_list:
        names = Path(image_list).read_text().splitlines()
        paths = [root / name for name in names]
    elif name_files:
        names = [str(n) for file in name_files for n in np.load(root / file)]
        paths = [root / name for name in names]
    else:
        paths = find_images([root])
    if unique:
        paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("Input image list is empty")
    # Dataset trees may intentionally contain symlinks to raw image storage.
    # Constrain declared names lexically; resolving every target would reject
    # valid dataset layouts such as ALTO's train -> raw/Train image links.
    for path in paths:
        try:
            relative = path.relative_to(root)
        except ValueError:
            raise ValueError(f"Image name escapes its dataset root: {path}") from None
        if ".." in relative.parts:
            raise ValueError(f"Image name escapes its dataset root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing input image: {path}")
    if limit is not None:
        limit_value = integer_tensor(limit, name="limit", minimum=1)
        if limit_value.ndim:
            raise ValueError("limit must be a scalar positive integer")
        limit = int(limit_value)
        indices = sorted(
            np.random.default_rng(seed).choice(
                len(paths), min(limit, len(paths)), replace=False
            )
        )
        paths = [paths[i] for i in indices]
    dataset = ImageList(paths, transform)
    dataset.input_root = root
    return dataset
