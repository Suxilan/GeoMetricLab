"""
DINOv2 特征可视化
"""
import os
import sys
sys.path.append('/home/sxl/project/MoSAICv2')
import dataclasses
from pathlib import Path
from typing import Tuple

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import random
import colorsys
import skimage
import torchvision
import torch.nn.functional as F

from torchvision import transforms
from matplotlib.patches import Polygon
from skimage.measure import find_contours

@dataclasses.dataclass
class VisAttentionConfig:
    """配置参数数据类"""
    image_path: str  # 输入图像路径
    output_dir: str  # 输出目录
    model_name: str = "dinov2_vitb14_reg"  # 模型名称
    layer: int | str = 11  # 特征提取层
    facet: str = "attn"  # 特征类型 (attn/key/value/token)
    threshold: float = 0.5  # 注意力阈值
    patch_size: int = 14  # 模型分块大小
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # 图像均值
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)  # 图像标准差
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 运行设备
    add_norm: bool = False  # 是否添加LayerNorm
    norm_descs: bool = False  # 是否归一化描述子
    dpi: int = 100  # 输出图像DPI
    blur: bool = True  # 是否模糊
    contour: bool = True  # 是否绘制轮廓
    alpha: float = 0.5  # 掩膜透明度

class DINOV2Wrapper:
    def __init__(self, config: VisAttentionConfig):
        self.config = config
        self.model = self._build_model()
        self.num_layers = self.model.n_blocks 
        self.feature_extractor = self._build_feature_extractor()

    def _build_model(self):
        """构建DINOv2模型"""
        model = torch.hub.load('third_party/dinov2', self.config.model_name, source='local')
        model.eval().to(self.config.device)
        return model
        
    def _build_feature_extractor(self):
        """支持多层特征提取"""
        if isinstance(self.config.layer, str) and self.config.layer.lower() == "all":
            return [self._build_single_extractor(l) for l in range(self.num_layers)]
        return self._build_single_extractor(self.config.layer)
    
    def _build_single_extractor(self, layer: int):
        """构建单层特征提取器"""
        return DinoV2ExtractFeatures(
            self.model,
            layer=layer,
            facet=self.config.facet,
            add_norm=self.config.add_norm,
            norm_descs=self.config.norm_descs,
            device=self.config.device
        )


class DinoV2ExtractFeatures:
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self, dino_model, 
                 layer: int, 
                 facet ="token", 
                 add_norm=True,
                norm_descs=True, device: str = "cpu") -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        self.dino_model = dino_model
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        self.add_norm = add_norm
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            # 对于注意力矩阵，直接保存输出
            self._hook_out = output
        return _forward_hook
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        B, _, H, W = img.shape
        with torch.no_grad():
            _ = self.dino_model(img)
            
            res = self._hook_out
            if self.facet == "token":
                if self.add_norm:
                    res = self.dino_model.norm(res)
                
            if self.facet in ["query", "key", "value", "attn"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2*d_len]
                elif self.facet == "value":
                    res = res[:, :, 2*d_len:]
                else:
                    num_heads = self.dino_model.blocks[self.layer].attn.num_heads
                    head_dim = d_len // num_heads
                    qkv = res.reshape(B, res.shape[1], 3, num_heads, head_dim)
                    q = qkv[:, :, 0]
                    k = qkv[:, :, 1]
                    scale = self.dino_model.blocks[self.layer].attn.scale
                    q = q.permute(0, 2, 1, 3)
                    k = k.permute(0, 2, 1, 3)
                    attn = q @ k.transpose(-2, -1) * scale
                    attn = attn.softmax(dim=-1)
                    res = attn
                               
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None   # Reset the hook
        
        return res
    
    def __del__(self):
        self.fh_handle.remove()

class ImageProcessor:
    """图像处理工具类"""
    @staticmethod
    def preprocess(image_path: str, 
                   img_mean: Tuple[float, float, float],
                   img_std: Tuple[float, float, float],
                   patch_size: int = 14,
                   device: str = "cpu") -> torch.Tensor:
        """图像预处理流程"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_scale = 10
        image = cv2.resize(image, (image.shape[1]//resize_scale, image.shape[0]//resize_scale))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, 
                                std=img_std),
            transforms.CenterCrop((image.shape[0]//patch_size * patch_size, image.shape[1]//patch_size * patch_size))
        ])
        image = transform(image).unsqueeze(0)
        return image.to(device)
        
    @staticmethod
    def denormalize(tensor: torch.Tensor, 
                    mean: Tuple[float, float, float],
                    std: Tuple[float, float, float]) -> torch.Tensor:
        """反归一化图像张量"""
        inverse_mean = [-m/s for m, s in zip(mean, std)]
        inverse_std = [1/s for s in std]
        return transforms.Normalize(mean=inverse_mean, std=inverse_std)(tensor)

class AttentionVisualizer:
    """注意力可视化引擎"""
    def __init__(self, config: VisAttentionConfig):
        self.config = config
        self.dino = DINOV2Wrapper(config)
        self.image_processor = ImageProcessor()
        
    def visualize(self):
        """执行完整可视化流程"""
        # 1. 预处理
        image_tensor = self.image_processor.preprocess(
            self.config.image_path,
            self.config.img_mean,
            self.config.img_std,
            self.config.patch_size,
            self.config.device
        )
        
        # 2. 提取特征
        if isinstance(self.dino.feature_extractor, list):
            for layer_idx, extractor in enumerate(self.dino.feature_extractor):
                self._process_single_layer(extractor, image_tensor, layer_idx)
        else:
            self._process_single_layer(self.dino.feature_extractor, image_tensor, self.config.layer)

    def _process_single_layer(self, extractor, image_tensor, layer_idx):
        """处理单层可视化"""
        with torch.no_grad():
            attentions = extractor(image_tensor)
            attentions = self._get_patch_tokens(attentions)
        self._process_attentions(attentions, image_tensor, layer_idx)
        
    def _get_patch_tokens(self, attentions: torch.Tensor):
        """获取特征图的patch tokens"""
        if "reg" in self.config.model_name:
            return attentions[0, :, 0, 5:]
        else:
            return attentions[0, :, 0, 1:]
    
    def _process_attentions(self, 
                           attentions: torch.Tensor,
                           image_tensor: torch.Tensor,
                           layer_idx: int):
        # 创建层级目录结构
        layer_dir = Path(self.config.output_dir) / f"layer{layer_idx}"
        attn_dir = layer_dir / "attn"
        mask_dir = layer_dir / "mask"
        attn_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        """处理注意力矩阵并生成可视化"""
        nh = attentions.shape[0]
        attention_org = attentions.clone()
        attentions = attentions.reshape(nh, -1)
        threshold = 0.5
        B, _, H, W = image_tensor.shape

        PATCH_SIZE = self.config.patch_size         # 模型分块大小

        patch_h = H // PATCH_SIZE
        patch_w = W // PATCH_SIZE

        images_plot = self.image_processor.denormalize(image_tensor, self.config.img_mean, self.config.img_std)
        combined_attn = torch.sum(attention_org, dim=0).reshape(patch_h, patch_w)

        if threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, patch_h, patch_w).float()
            # interpolate
            th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest")[0].cpu().numpy()
            
            combined_attn_flat = combined_attn.flatten()
            val, idx = torch.sort(combined_attn_flat)
            val /= torch.sum(val)
            cumval = torch.cumsum(val, dim=0)
            th_combined_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            th_combined_attn = th_combined_attn[idx2].reshape(patch_h, patch_w).float()
            th_combined_attn = F.interpolate(th_combined_attn.unsqueeze(0).unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest")[0, 0].cpu().numpy()

        attentions = attentions.reshape(nh, patch_h, patch_w)
        attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=PATCH_SIZE, mode="nearest")[0].cpu().detach().numpy()

        # save attentions heatmaps
        torchvision.utils.save_image(torchvision.utils.make_grid(images_plot, normalize=True, scale_each=True), os.path.join(self.config.output_dir, "img.png"))
        for j in range(nh):
            fname = os.path.join(attn_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")
            
        combined_attn = F.interpolate(combined_attn.unsqueeze(0).unsqueeze(0), 
                                      scale_factor=PATCH_SIZE, 
                                      mode="nearest")[0, 0].cpu().detach().numpy()
        fname = os.path.join(attn_dir, "attn-combined.png")
        plt.imsave(fname=fname, arr=combined_attn, format='png')
        print(f"{fname} saved.")

        if threshold is not None:
            image = skimage.io.imread(os.path.join(self.config.output_dir, "img.png"))
            for j in range(nh):
                self.display_instances(image, th_attn[j], fname=os.path.join(mask_dir, "mask_th" + str(threshold) + "_head" + str(j) +".png"), config=self.config)
            self.display_instances(image, th_combined_attn, fname=os.path.join(mask_dir, "mask_th" + str(threshold) + "_combined.png"), config=self.config)
            
    def _apply_mask(self, image, mask, color, alpha=0.5):
        for c in range(3):
            image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        return image
    
    def _random_colors(self, N, bright=True):
        """
        Generate random colors.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    
    def display_instances(self, image, mask, fname="test", config: VisAttentionConfig=None):
        # 获取图像原始尺寸
        height, width = image.shape[:2]
        
        figsize = (width / config.dpi, height / config.dpi)  # 根据 dpi 转换为英寸
        
        # 创建 Figure 和 Axes
        fig = plt.figure(figsize=figsize, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])  # Axes 占满整个 Figure
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # 处理掩膜和绘制
        N = 1
        mask = mask[None, :, :]
        colors = self._random_colors(N)
        
        # 设置坐标轴范围和关闭坐标轴
        ax.set_ylim(height, 0)  # 修复方向（避免图像上下颠倒）
        ax.set_xlim(0, width)
        ax.axis('off')
        
        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
            _mask = mask[i]
            if config.blur:
                _mask = cv2.blur(_mask, (10, 10))
            # 应用遮罩
            masked_image = self._apply_mask(masked_image, _mask, color, config.alpha)
            # 绘制轮廓
            if config.contour:
                padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = _mask
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    verts = np.fliplr(verts) - 1
                    p = Polygon(verts, facecolor="none", edgecolor=color)
                    ax.add_patch(p)
        
        # 确保图像按原始比例显示
        ax.imshow(masked_image.astype(np.uint8), aspect='equal')  
        
        # 保存时去除空白边距
        fig.savefig(fname, bbox_inches='tight', pad_inches=0, dpi=config.dpi)
        plt.close()
        print(f"{fname} saved.")
        return

def main(config: VisAttentionConfig):
    """主流程"""
    # 初始化输出目录
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 执行可视化
    visualizer = AttentionVisualizer(config)
    visualizer.visualize()

if __name__ == "__main__":
    # 配置初始化
    config = VisAttentionConfig(
        image_path="GF1_WFV3_E112.1_N32.3_20240514_L1A13404954001.jpeg.jpg",
        output_dir="viz_scripts/dv2g_reg_attn_map",
        model_name="dinov2_vitg14_reg",
        layer="all",
        facet="attn",
        threshold=0.5
    )
    
    # 执行主程序
    main(config)