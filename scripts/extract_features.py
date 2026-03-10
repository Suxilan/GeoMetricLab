"""Extract image features recursively and save to HDF5 cache.

Usage examples:
    python scripts/extract_features_for_init_whitening.py \
        --model resnet50_gem \
        --roots data/GL3D/train \
        --batch_size 64  \
        --out_dir cache \
        --img_size 384
"""
from __future__ import annotations

import argparse
import os, sys
from pathlib import Path
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T2
from PIL import Image

# project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.geoencoder import GeoEncoder
from config.model_hubs import MODEL_HUBS
from weights.util import load_weights, get_weight_config
from src.utils.logger import print_rank_0
from src.utils.io import gather_distributed_predictions, save_features_h5
import pytorch_lightning as pl
import torch.nn as nn

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(roots: List[Path], exts=IMG_EXTS) -> List[Path]:
    imgs = []
    for r in roots:
        r = Path(r)
        if not r.exists():
            print_rank_0(f"Warning: root {r} does not exist, skipping")
            continue
        for p in r.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                imgs.append(p)
    imgs.sort()
    return imgs


class ImageDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        # return image tensor and index so we can recover original ordering
        return img, idx

def load_model_and_weights(args) -> nn.Module:
    # Determine weight path and config
    weight_path = None
    model_config = None

    if args.model is not None:
        if args.model in MODEL_HUBS:
            model_config = MODEL_HUBS[args.model]
        else:
            raise SystemExit(f"Error: No model config found for '{args.model}'. Available models: {list(MODEL_HUBS.keys())}")
        
    if args.weights is not None:
        # Resolve weight path
        weight_path, model_config = get_weight_config(args.weights)

        if model_config is None:
            raise SystemExit(f"Error: No config found for {args.weights}.")

        weight_path = weight_path or model_config.get("weight_path")
        if weight_path is None:
            raise SystemExit(f"Hub entry '{args.weights}' has no weight_path; cannot evaluate without weights.")
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise SystemExit(f"Weight file not found for '{args.weights}': {weight_path}")
        
    if model_config is None:
        raise SystemExit("Error: No model config provided. Use --model or --weights to specify model configuration.")
    
    print_rank_0(f"Model Config: {model_config}")
    print_rank_0(f"Weights: {weight_path}")
    
    # Setup model
    model = GeoEncoder(
        backbone_name=model_config['backbone'],
        aggregator_name=model_config['aggregator'],
        backbone_args=model_config.get('backbone_args', {}),
        aggregator_args=model_config.get('aggregator_args', {}),
        whitening=model_config.get("whitening", False),
        whitening_dim=model_config.get("whitening_dim", 256),
    )
    
    # Load weights
    if weight_path:
        print_rank_0(f"Loading weights...")
        load_weights(model, str(weight_path), args.weights)

    return model, weight_path, model_config

def extract_and_save_with_trainer(
        model: nn.Module, 
        paths: List[Path], 
        out_h5: Path, 
        batch_size: int = 128, 
        devices: str | int = "auto",
        strategy: str | None = None, 
        num_workers: int = 4, 
        img_size: int = 224
    ):
    transform = T2.Compose([
        T2.ToImage(),
        T2.Resize(size=img_size, interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class ExtractModule(pl.LightningModule):
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            images, indices = batch
            images = images.to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                feats = outputs[0] if isinstance(outputs, tuple) else outputs
                feats = feats.detach().cpu()
            return {"features": feats, "indices": indices.cpu()}

    trainer = pl.Trainer(
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=devices,
        strategy=strategy,
        logger=False,
        enable_progress_bar=True,
    )

    module = ExtractModule(model)
    raw_preds = trainer.predict(module, dataloaders=loader)

    feats, idxs = gather_distributed_predictions(raw_preds, trainer)

    # Sort by original index
    order = torch.argsort(idxs)
    sorted_feats = feats[order].numpy()
    sorted_idxs = idxs[order].numpy()

    image_names = [str(paths[int(i)]) for i in sorted_idxs]

    # Only rank 0 should write the cache file
    if getattr(trainer, "is_global_zero", False):
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        save_features_h5(sorted_feats, image_names, out_h5.name, cache_dir=out_h5.parent)
        print_rank_0(f"Saved features for {len(paths)} images to {out_h5}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None, help="Path or hub key for weights")
    parser.add_argument('--model', type=str, default=None, help="Model config")
    parser.add_argument("--roots", nargs="+", required=True, help="One or more root directories to scan (recursive)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out_dir", type=str, default="cache")
    parser.add_argument("--prefix", type=str, default="features")
    parser.add_argument("--devices", type=str, default='auto' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--strategy", type=str, default='auto' if torch.cuda.device_count() > 1 else None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()
    roots = [Path(r) for r in args.roots]
    paths = find_images(roots)
    if len(paths) == 0:
        print_rank_0("No images found under provided roots.")
        return

    # Determine weight path and config
    model, weight_path, model_config = load_model_and_weights(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # model_name in filename (use weight stem if available else model key)
    model_name = Path(weight_path).stem if weight_path else args.model
    out_h5 = out_dir / f"{args.prefix}_{model_name}_{len(paths)}.h5"

    extract_and_save_with_trainer(model, paths, out_h5, batch_size=args.batch_size, devices=args.devices, strategy=args.strategy, num_workers=args.num_workers, img_size=args.img_size)


if __name__ == "__main__":
    main()
