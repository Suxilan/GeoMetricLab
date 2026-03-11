#!/usr/bin/env python3
"""Build retrieval-based image pair lists for SfM / COLMAP pipelines.

Given a scene root, model configuration, and top-k, this script:
1. Recursively finds images under the scene root.
2. Extracts global descriptors with GeoEncoder.
3. Retrieves top-k neighbors for each image.
4. Writes a COLMAP-compatible pair list file where each line is:
   ``image_name_1 image_name_2``

Example:
    python scripts/build_sfm_pairs.py \
        --root data/1dsfm/images.Trafalgar/Trafalgar \
        --weights peft_dinov2_scpp_whiten1536_gl3d_pretrain \
        --topk 20 \
        --output results/pairs/trafalgar_top20.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.model_hubs import MODEL_HUBS
from src.models.geoencoder import GeoEncoder
from src.utils.io import load_features_h5, save_features_h5
from src.utils.logger import print_rank_0
from weights.util import get_weight_config, load_weights


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a COLMAP pair list from retrieval results.")
    parser.add_argument("--root", type=str, required=True, help="Scene root containing images (recursive search).")
    parser.add_argument("--weights", type=str, default=None, help="Weight hub key or weight path.")
    parser.add_argument("--model", type=str, default=None, help="Model hub key when evaluating without pretrained weights.")
    parser.add_argument("--topk", type=int, default=20, help="Number of retrieved neighbors per image before deduplication.")
    parser.add_argument("--img_size", type=int, default=518, help="Inference resize.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for descriptor extraction.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--output", type=str, required=True, help="Output COLMAP pair list path.")
    parser.add_argument("--cache", type=str, default=None, help="Optional H5 cache path for descriptors.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device.")
    parser.add_argument("--prefer_gpu_faiss", action="store_true", help="Use FAISS GPU index when available.")
    return parser.parse_args()


def find_images(root: Path, exts: Iterable[str] = IMG_EXTS) -> List[Path]:
    image_paths: List[Path] = []
    valid_exts = {ext.lower() for ext in exts}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_exts:
            image_paths.append(path)
    image_paths.sort()
    return image_paths


def build_transform(img_size: int) -> T2.Compose:
    return T2.Compose(
        [
            T2.ToImage(),
            T2.Resize(size=(img_size, img_size), interpolation=T2.InterpolationMode.BICUBIC, antialias=True),
            T2.ToDtype(torch.float32, scale=True),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SceneImageDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform: T2.Compose):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, index


def load_model_and_weights(args: argparse.Namespace) -> tuple[nn.Module, Path | None, dict]:
    weight_path = None
    model_config = None

    if args.model is not None:
        if args.model not in MODEL_HUBS:
            raise SystemExit(f"Error: unknown model '{args.model}'. Available models: {list(MODEL_HUBS.keys())}")
        model_config = MODEL_HUBS[args.model]

    if args.weights is not None:
        weight_path, model_config = get_weight_config(args.weights)
        if model_config is None:
            raise SystemExit(f"Error: no config found for {args.weights}.")
        weight_path = weight_path or model_config.get("weight_path")
        if weight_path is None:
            raise SystemExit(f"Hub entry '{args.weights}' has no weight_path.")
        weight_path = Path(weight_path)
        if not weight_path.exists():
            raise SystemExit(f"Weight file not found: {weight_path}")

    if model_config is None:
        raise SystemExit("Error: provide --model or --weights.")

    model = GeoEncoder(
        backbone_name=model_config["backbone"],
        aggregator_name=model_config["aggregator"],
        backbone_args=model_config.get("backbone_args", {}),
        aggregator_args=model_config.get("aggregator_args", {}),
        whitening=model_config.get("whitening", False),
        whitening_dim=model_config.get("whitening_dim", 256),
    )

    if weight_path is not None:
        print_rank_0(f"Loading weights from {weight_path}...")
        load_weights(model, str(weight_path), args.weights)

    return model, weight_path, model_config


@torch.inference_mode()
def extract_features(
    model: nn.Module,
    image_paths: Sequence[Path],
    batch_size: int,
    num_workers: int,
    img_size: int,
    device: str,
) -> torch.Tensor:
    dataset = SceneImageDataset(image_paths, transform=build_transform(img_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = model.to(device)
    model.eval()

    feature_chunks: list[torch.Tensor] = [torch.empty(0)] * len(dataset)
    for images, indices in tqdm(dataloader, desc="Extracting features", unit="batch"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        features = outputs[0] if isinstance(outputs, tuple) else outputs
        features = features.detach().cpu()
        for feature, index in zip(features, indices.tolist()):
            feature_chunks[index] = feature

    return torch.stack(feature_chunks, dim=0)


def maybe_load_cache(cache_path: Path) -> tuple[torch.Tensor, list[str]]:
    data = load_features_h5(cache_path.name, cache_dir=cache_path.parent)
    features = torch.from_numpy(np.asarray(data["features"], dtype=np.float32))
    image_names = list(data["image_names"])
    return features, image_names


def maybe_save_cache(cache_path: Path, features: torch.Tensor, image_names: list[str]) -> None:
    save_features_h5(features.numpy(), image_names, cache_path.name, cache_dir=cache_path.parent)
    print_rank_0(f"Saved feature cache to {cache_path}")


def faiss_topk_neighbors(features: torch.Tensor, topk: int, prefer_gpu: bool) -> np.ndarray:
    import faiss

    descriptors = F.normalize(features.float(), p=2, dim=-1).cpu().numpy().astype(np.float32, copy=False)
    faiss.normalize_L2(descriptors)

    dimension = descriptors.shape[1]
    k_search = min(topk + 1, descriptors.shape[0])
    index_cpu = faiss.IndexFlatIP(dimension)
    index = index_cpu

    if prefer_gpu and hasattr(faiss, "StandardGpuResources"):
        try:
            if faiss.get_num_gpus() > 0:
                resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(resources, 0, index_cpu)
        except Exception as exc:
            print_rank_0(f"FAISS GPU unavailable, fallback to CPU: {exc}")
            index = index_cpu

    index.add(descriptors)
    _, indices = index.search(descriptors, k_search)
    return indices


def build_colmap_pairs(relative_names: Sequence[str], neighbor_indices: np.ndarray, topk: int) -> list[tuple[str, str]]:
    seen_pairs: set[tuple[str, str]] = set()
    pair_list: list[tuple[str, str]] = []

    for query_index, neighbors in enumerate(neighbor_indices):
        kept = 0
        query_name = relative_names[query_index]
        for neighbor_index in neighbors:
            if neighbor_index == query_index:
                continue
            ref_name = relative_names[int(neighbor_index)]
            pair = tuple(sorted((query_name, ref_name)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pair_list.append(pair)
            kept += 1
            if kept >= topk:
                break

    return pair_list


def main() -> None:

    args = parse_args()
    print_rank_0("==== SfM Pair List构建任务启动 ====")
    print_rank_0(f"参数: root={args.root}, weights={args.weights}, model={args.model}, topk={args.topk}, img_size={args.img_size}, batch_size={args.batch_size}, output={args.output}, cache={args.cache}, device={args.device}")
    if args.topk <= 0:
        raise SystemExit("Error: --topk must be positive.")

    scene_root = Path(args.root)
    if not scene_root.exists():
        raise SystemExit(f"Error: scene root does not exist: {scene_root}")

    print_rank_0("[1] 正在递归查找图片...")
    image_paths = find_images(scene_root)
    if not image_paths:
        raise SystemExit(f"Error: no images found under {scene_root}")
    print_rank_0(f"找到 {len(image_paths)} 张图片于 {scene_root}")
    relative_names = [str(path.relative_to(scene_root)).replace(os.sep, "/") for path in image_paths]

    if args.cache:
        cache_path = Path(args.cache)
    else:
        model_tag = args.weights or args.model or "model"
        scene_tag = scene_root.name
        cache_path = Path("cache") / "sfm_pairs" / f"{scene_tag}_{Path(model_tag).stem}_{len(image_paths)}.h5"

    if cache_path.exists():
        print_rank_0(f"[2] 加载特征缓存: {cache_path}")
        features, cached_names = maybe_load_cache(cache_path)
        if cached_names != relative_names:
            raise SystemExit("Error: cache image ordering does not match current scene file ordering.")
    else:
        print_rank_0("[2] 加载模型与权重...")
        model, _, _ = load_model_and_weights(args)
        print_rank_0("[3] 提取图片特征...")
        features = extract_features(
            model=model,
            image_paths=image_paths,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            device=args.device,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        maybe_save_cache(cache_path, features, relative_names)

    print_rank_0("[4] 检索Top-K邻居...")
    neighbor_indices = faiss_topk_neighbors(features, topk=args.topk, prefer_gpu=args.prefer_gpu_faiss)
    print_rank_0("[5] 构建COLMAP配对列表...")
    pair_list = build_colmap_pairs(relative_names, neighbor_indices, topk=args.topk)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for left_name, right_name in pair_list:
            file.write(f"{left_name} {right_name}\n")

    print_rank_0(f"[6] 写入 {len(pair_list)} 个唯一配对到 {output_path}")
    print_rank_0("==== SfM Pair List任务完成 ====")


if __name__ == "__main__":
    main()