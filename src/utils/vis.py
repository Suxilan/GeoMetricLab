import math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Up to 32 distinct colors for categorical assignment maps
_ASSIGN_PALETTE_32 = np.array(
    [
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207], [174, 199, 232], [255, 187, 120],
        [152, 223, 138], [255, 152, 150], [197, 176, 213], [196, 156, 148],
        [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229],
        [57, 59, 121], [82, 84, 163], [107, 110, 207], [156, 158, 222],
        [99, 121, 57], [140, 162, 82], [181, 207, 107], [206, 219, 156],
        [140, 109, 49], [189, 158, 57], [231, 186, 82], [231, 203, 148],
    ],
    dtype=np.uint8,
)


def tensor_to_rgb_pil(img: torch.Tensor) -> Image.Image:
    """(3,H,W) normalized tensor -> PIL RGB (ImageNet de-normalize)."""
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected (3,H,W) tensor, got {tuple(img.shape)}")

    x = img.detach().float().cpu()
    x = x * _IMAGENET_STD + _IMAGENET_MEAN
    x = x.clamp(0.0, 1.0)
    x = (x * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x, mode="RGB")


def aux_to_overlay_pil(
    aux: object,
    out_size: tuple[int, int],
    key: str = "assign_map",
    cmap: str = "jet",
) -> Image.Image | tuple[Image.Image, Image.Image] | None:
    """Convert aux tensors to a visual overlay image (same size as RGB).

    Single responsibility: take aux output (dict or tensor) and return an RGB heatmap
    image that can be alpha-blended on top of the RGB input.

        Default behavior:
        - key='assign_map': expects (K,h,w) (soft assignment). Returns a tuple:
            (confidence heatmap, categorical label map)

    Extensible:
    - You can pass other keys like 'attn' and provide tensors shaped (H,W), (heads,H,W), etc.
      This function will try to reduce to (H,W) by mean over leading dims.
    """
    t: torch.Tensor | None = None
    if isinstance(aux, dict):
        v = aux.get(key, None)
        if torch.is_tensor(v):
            t = v
    elif torch.is_tensor(aux):
        t = aux

    if t is None:
        return None

    # Accept (B, ...) by taking first item
    # Only strip batch dimension for 4D tensors to preserve (K,H,W) when K=1
    if t.dim() == 4 and t.size(0) == 1:
        t = t[0]

    x = t.detach().float().cpu()

    # Reduce to (H,W)
    if key == "assign_map":
        # (K,h,w) -> max over K (confidence)
        if x.dim() != 3:
            return None
        x2d = x.max(dim=0).values
        labels = x.argmax(dim=0).to(torch.int64).numpy()
        labels = np.asarray(labels, dtype=np.int64) % _ASSIGN_PALETTE_32.shape[0]
        label_rgb = _ASSIGN_PALETTE_32[labels]
        label_img = Image.fromarray(label_rgb, mode="RGB")
        label_img = label_img.resize(out_size, resample=Image.Resampling.NEAREST)
    elif key == "discard_map":
        # (1,h,w) (or h,w) -> (h,w)
        if x.dim() == 3:
            x2d = x[0] if x.size(0) == 1 else x.mean(dim=0)
        elif x.dim() == 2:
            x2d = x
        else:
            return None
    else:
        # generic: if (H,W) use as-is; if (>2 dims) mean over leading dims
        if x.dim() == 2:
            x2d = x
        elif x.dim() > 2: 
            x2d = x.mean(dim=tuple(range(0, x.dim() - 2)))
        else:
            return None

    m = x2d.numpy()
    m = m - np.nanmin(m)
    denom = np.nanmax(m)
    if not np.isfinite(denom) or denom <= 1e-12:
        m = np.zeros_like(m, dtype=np.float32)
    else:
        m = (m / denom).astype(np.float32)

    cm = plt.get_cmap(cmap)
    rgba = cm(m)  # (h,w,4) float in [0,1]
    rgb = (rgba[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    heat = Image.fromarray(rgb, mode="RGB")
    heat = heat.resize(out_size, resample=Image.Resampling.BILINEAR)
    if key == "assign_map":
        return heat, label_img
    return heat

def plot_single_image(ax, img_path_or_obj, title, border_color=None):
    """辅助函数：在指定的 ax 上画一张图"""
    if isinstance(img_path_or_obj, Image.Image):
        img = img_path_or_obj
    else:
        img = Image.open(img_path_or_obj)
    
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if border_color:
        for sp in ax.spines.values():
            sp.set_edgecolor(border_color)
            sp.set_linewidth(3)

def visual_grid(query_path, gallery_imgs, gallery_titles, gallery_colors, grid, query_title="Query"):
    """
    绘制 Query (左) + TopK Gallery (右侧网格)
    """
    rows_g, cols_g = grid
    K = len(gallery_imgs)
    assert rows_g * cols_g == K, f"Grid {grid} does not match gallery count {K}"

    fig_cols = cols_g + 1  # 左侧留一列给 Query
    fig_rows = rows_g
    
    plt.figure(figsize=(3.5 * fig_cols, 3.5 * fig_rows))

    # --- 1. 画 Query (放在第1行第1列) ---
    ax_q = plt.subplot(fig_rows, fig_cols, 1)
    plot_single_image(ax_q, query_path, query_title, border_color="black")

    # --- 2. 画 Gallery ---
    for i, (img, title, color) in enumerate(zip(gallery_imgs, gallery_titles, gallery_colors)):
        # 计算子图位置：跳过每一行的第1列(留给Query的空白，或者只在第一行画Query)
        # 这里采用简单布局：Query 占 (0,0)，Gallery 填满剩下的格子
        # 逻辑：Gallery 的第 i 个图，对应的网格坐标 (r, c)
        r = i // cols_g
        c = (i % cols_g) + 1  # +1 是因为第0列是Query
        
        subplot_idx = r * fig_cols + (c + 1)
        ax = plt.subplot(fig_rows, fig_cols, subplot_idx)
        plot_single_image(ax, img, title, border_color=color)

    plt.tight_layout()
    plt.show()

def visual_gt(query_path, gt_imgs, gt_titles, gt_colors, query_title="Query", max_cols=5):
    """
    绘制 Query (左) + 所有 Ground Truth (右侧自适应网格)
    """
    total_gt = len(gt_imgs)
    if total_gt == 0:
        print("No Ground Truth to visualize.")
        return

    # 计算布局
    cols_gt = min(max_cols, total_gt)
    rows_gt = math.ceil(total_gt / cols_gt)
    
    fig_cols = cols_gt + 1 # 左侧留一列给 Query
    fig_rows = rows_gt

    plt.figure(figsize=(3.5 * fig_cols, 3.5 * fig_rows))
    plt.suptitle("Ground Truth Check")

    # --- 1. 画 Query ---
    ax_q = plt.subplot(fig_rows, fig_cols, 1)
    plot_single_image(ax_q, query_path, query_title, border_color="black")

    # --- 2. 画 GT ---
    for i, (img, title, color) in enumerate(zip(gt_imgs, gt_titles, gt_colors)):
        r = i // cols_gt
        c = (i % cols_gt) + 1
        
        subplot_idx = r * fig_cols + (c + 1)
        ax = plt.subplot(fig_rows, fig_cols, subplot_idx)
        plot_single_image(ax, img, title, border_color=color)

    plt.tight_layout()
    plt.show()
