"""Group image rows by explicit modality without changing their identity order."""

import torch

from .identities import integer_tensor


def modality_collate(rows: list[dict]) -> dict:
    """Collate rows with images [C,H,W], modality, and optional label/index.

    Each modality must have a consistent image shape, but shapes and counts may
    differ between modalities. positions maps each grouped image to its original
    row. labels and index remain in original row order. No pairing is inferred
    from positions; labels must already share the Dataset's identity namespace.
    """
    if not rows:
        raise ValueError("Cannot collate an empty image collection")
    images = {}
    positions = {}
    modalities = []
    for position, row in enumerate(rows):
        modality = row["modality"]
        if not isinstance(modality, str) or not modality:
            raise ValueError("Each image needs an explicit nonempty modality")
        image = row["images"]
        if not isinstance(image, torch.Tensor) or image.ndim != 3:
            raise ValueError("Each transformed image must be a Tensor [C,H,W]")
        images.setdefault(modality, []).append(image)
        positions.setdefault(modality, []).append(position)
        modalities.append(modality)
    grouped_images = {}
    for modality, group in images.items():
        if any(image.shape != group[0].shape for image in group):
            raise ValueError(f"Inconsistent image shapes within modality {modality}")
        grouped_images[modality] = torch.stack(group)
    result = {
        "images": grouped_images,
        "positions": {
            modality: torch.tensor(indices, dtype=torch.long)
            for modality, indices in positions.items()
        },
        "modalities": modalities,
    }
    for source, target in (("label", "labels"), ("index", "index")):
        present = [source in row for row in rows]
        if any(present) and not all(present):
            raise ValueError(f"{source} must be present for every image or none")
        if all(present):
            values = [row[source] for row in rows]
            tensor = integer_tensor(values, name=source)
            if tensor.ndim != 1:
                raise ValueError(f"{source} must contain integer scalar IDs")
            result[target] = tensor
    identities_present = ["sample_id" in row for row in rows]
    if any(identities_present) and not all(identities_present):
        raise ValueError("sample_id must be present for every image or none")
    if all(identities_present):
        result["sample_ids"] = [row["sample_id"] for row in rows]
    return result


# Public collator output layout, consumed by training assembly.
modality_collate.batch_format = "modality_id"
