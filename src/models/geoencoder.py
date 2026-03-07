import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ResNet, DinoV2, PEFTDinoV2, DinoV3, SwinTransformer, ConvNeXt
from .aggregator import (
    Avg, CLS,
    GeM, SuperGeM , 
    CGeM,
    DGeM,
    BoQ, Salad,
    NetVLAD, GhostVLAD, SuperVLAD,
    CFP,SoftPCFP,
    EigenPlace,CosPlace,
    RegVPR,
    SCPP,
    CricaVPR,
)
from src.utils.logger import print_rank_0

class GeoEncoder(nn.Module):
    def __init__(self, 
        backbone_name, 
        aggregator_name, 
        backbone_args={}, 
        aggregator_args={},
        whitening: bool = False,
        whitening_dim: int = 256, # only used if whitening=True
        final_norm: bool = True,
        use_bn: bool = True,
    ):
        super().__init__()
        
        # 1. Initialize Backbone
        self.backbone_name = backbone_name.lower()
        self.backbone_args = backbone_args
        print_rank_0(f"[{__class__.__name__}] Initializing Backbone: {self.backbone_name}")
        print_rank_0(f"[{__class__.__name__}] Backbone Args: {backbone_args}")

        if 'resnet' in self.backbone_name or 'resnext' in self.backbone_name:
            self.backbone = ResNet(backbone_name=backbone_name, **backbone_args)
        elif 'peft_dinov2' in self.backbone_name:
            self.backbone = PEFTDinoV2(backbone_name=backbone_name, **backbone_args)
        elif 'dinov2' in self.backbone_name:
            self.backbone = DinoV2(backbone_name=backbone_name, **backbone_args)
        elif 'convnext' in self.backbone_name:
            self.backbone = ConvNeXt(backbone_name=backbone_name, **backbone_args)
        elif 'dinov3' in self.backbone_name:
            self.backbone = DinoV3(backbone_name=backbone_name, **backbone_args)
        elif 'swin' in self.backbone_name:
            self.backbone = SwinTransformer(backbone_name=backbone_name, **backbone_args)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        self.backbone_out_channels = self.backbone.out_channels
        
        # 2. Initialize Aggregator
        self.aggregator_name = aggregator_name.lower()
        self.aggregator_args = aggregator_args
        print_rank_0(f"[{__class__.__name__}] Initializing Aggregator: {self.aggregator_name}")
        print_rank_0(f"[{__class__.__name__}] Aggregator Args: {aggregator_args}")
        
        # Some aggregators need input channels
        if self.aggregator_name == 'avg':
            self.aggregator = Avg(in_channels=self.backbone_out_channels)
        elif self.aggregator_name == 'gem':
            self.aggregator = GeM(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'dgem':
            self.aggregator = DGeM(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'cgem':
            self.aggregator = CGeM(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'scpp':
            self.aggregator = SCPP(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'supergem':
            self.aggregator = SuperGeM(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'cosplace':
            self.aggregator = CosPlace(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'boq':
            self.aggregator = BoQ(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'salad':
            self.aggregator = Salad(num_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'cls':
            self.aggregator = CLS(in_channels=self.backbone_out_channels)
        elif self.aggregator_name == 'netvlad':
            self.aggregator = NetVLAD(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'supervlad':
            self.aggregator = SuperVLAD(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'ghostvlad':
            self.aggregator = GhostVLAD(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'cfp':
            self.aggregator = CFP(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'softp':
            self.aggregator = SoftPCFP(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'eigenplace':
            self.aggregator = EigenPlace(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'regvpr':
            self.aggregator = RegVPR(in_channels=self.backbone_out_channels, **aggregator_args)
        elif self.aggregator_name == 'cricavpr':
            self.aggregator = CricaVPR(in_channels=self.backbone_out_channels, **aggregator_args)
        else:
            raise ValueError(f"Unsupported aggregator: {aggregator_name}")
        self.out_channels = self.aggregator.out_channels

        self.use_bn = bool(use_bn)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_channels, affine=True)
            self.bn.apply(self._init_bn)
        else:
            self.bn = nn.Identity()
        # If finetune_whiten is requested, force-enable whitening and later
        self.whitening = whitening
        if self.whitening:
            if whitening_dim > self.out_channels:
                raise ValueError(
                    f"whitening_dim ({whitening_dim}) cannot be greater than aggregator output channels ({self.out_channels})"
                )
            print_rank_0(f"[{__class__.__name__}] Adding Whitening layer with output dim: {whitening_dim}")
            self.whitening_layer = nn.Linear(self.out_channels, whitening_dim, bias=True)
            self.out_channels = whitening_dim
        else:
            self.whitening_layer = nn.Identity()

        self.final_norm = final_norm
    
    def _init_bn(self, module):
        if isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    @torch.no_grad()
    def init_whitening(self, X: torch.Tensor, eps: float = 1e-4):
        """
        X: (N, D)  用于统计的特征（建议来自与你实际 forward 一致的路径/状态）
        eps: 协方差对角抖动，越大越稳但越“不过白”
        """
        device = self.whitening_layer.weight.device
        D_in = self.whitening_layer.in_features
        D_out = self.whitening_layer.out_features

        X = X.to(device).float()   # ✅ 保证 dtype/device 统一
        assert X.dim() == 2 and X.size(1) == D_in, f"X should be (N, {D_in}), got {tuple(X.shape)}"

        N = X.size(0)

        # 1) 高精度统计
        X64 = X.double()
        mean = X64.mean(dim=0)         # (D_in,)
        Xc = X64 - mean                # (N, D_in)

        # 2) 协方差 + ridge
        cov = (Xc.t() @ Xc) / max(N - 1, 1)
        cov = cov + eps * torch.eye(D_in, device=device, dtype=torch.float64)

        # 3) 特征分解（S 升序）
        S, U = torch.linalg.eigh(cov)

        # 4) 取 top-D_out
        idx = torch.argsort(S, descending=True)[:D_out]
        S = S[idx]
        U = U[:, idx]  # (D_in, D_out)

        # 5) whitening 矩阵：W = U * S^{-1/2}
        whitening_matrix = U * torch.rsqrt(S).unsqueeze(0)   # (D_in, D_out)

        new_weight = whitening_matrix.t().float()            # (D_out, D_in)
        new_bias = (-(mean @ whitening_matrix)).float()      # (D_out,)

        # 6) 写回参数
        self.whitening_layer.weight.data.copy_(new_weight)
        self.whitening_layer.bias.data.copy_(new_bias)

        print_rank_0("[*] Whitening computed and set on rank0.")

    def forward(self, x):
        # 1. Backbone Forward
        x = self.backbone(x)
        
        # 2. Handle Output Format
        # CNN: x is Tensor (B, C, H, W)
        # ViT: x is Tuple (cls, patch, attn)
        
        if isinstance(x, tuple):
            cls_token, patch_tokens, attn = x
            
            # Dispatch to aggregator
            if self.aggregator_name == 'cls':
                x, aux_info = self.aggregator(patch_tokens, cls_token=cls_token)
            elif self.aggregator_name == 'salad':
                x, aux_info = self.aggregator(patch_tokens, cls_token=cls_token)
            elif self.aggregator_name == 'regvpr':
                x, aux_info = self.aggregator(patch_tokens, cls_token=cls_token)
            elif self.aggregator_name == 'cricavpr':
                x, aux_info = self.aggregator(patch_tokens, cls_token=cls_token)
            elif self.aggregator_name == 'dighostvlad':
                x, aux_info = self.aggregator(patch_tokens, attn=attn)
            else:
                # For Avg, GeM, CosPlace, BoQ, NetVLAD -> Use patch tokens as feature map
                x, aux_info = self.aggregator(patch_tokens)
        else:
            # CNN case
            if self.aggregator_name == 'cls':
                raise ValueError("CLS cannot be used with CNN backbones (no CLS token).")
            elif self.aggregator_name == 'salad':
                raise ValueError("SALAD cannot be used with CNN backbones (no CLS token).")
            elif self.aggregator_name == 'regvpr':
                raise ValueError("RegVPR cannot be used with CNN backbones (no CLS token).")
            elif self.aggregator_name == 'cricavpr':
                raise ValueError("CricaVPR cannot be used with CNN backbones (no CLS token).")
            else:
                x, aux_info = self.aggregator(x)
        
        x = self.bn(x)
        # Apply Whitening if enabled
        x = self.whitening_layer(x)
        # 3. Final L2 Normalization
        if self.final_norm:
            x = F.normalize(x, p=2, dim=-1)
        
        return x, aux_info