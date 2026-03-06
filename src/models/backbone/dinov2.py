import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from src.utils.logger import print_rank_0

class DinoV2(nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14',
        'dinov2_vits14_reg',
        'dinov2_vitb14_reg',
        'dinov2_vitl14_reg',
        'dinov2_vitg14_reg'
    ]
    
    ARCH_CONFIG = {
        'dinov2_vits14': {'depth': 12, 'embed_dim': 384, 'patch_size': 14},
        'dinov2_vitb14': {'depth': 12, 'embed_dim': 768, 'patch_size': 14},
        'dinov2_vitl14': {'depth': 24, 'embed_dim': 1024, 'patch_size': 14},
        'dinov2_vitg14': {'depth': 40, 'embed_dim': 1536, 'patch_size': 14},
        # Reg variants have same depth/dim
        'dinov2_vits14_reg': {'depth': 12, 'embed_dim': 384, 'patch_size': 14},
        'dinov2_vitb14_reg': {'depth': 12, 'embed_dim': 768, 'patch_size': 14},
        'dinov2_vitl14_reg': {'depth': 24, 'embed_dim': 1024, 'patch_size': 14},
        'dinov2_vitg14_reg': {'depth': 40, 'embed_dim': 1536, 'patch_size': 14},
    }

    def print(self, msg):
        print_rank_0(f"[{self.__class__.__name__}] {msg}")
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        num_unfrozen_blocks=0,
        num_layers=None,
    ):
        """DinoV2 backbone with the ability to keep only the last num_unfrozen_blocks trainable.

        Args:
            backbone_name (str, optional): DinoV2 variant. Defaults to "dinov2_vitb14".
            num_unfrozen_blocks (int, optional): number of blocks to unfreeze. Defaults to 0.
            num_layers (int, optional): number of layers to run. If None, run all layers. Defaults to None.

        Raises:
            ValueError: if the backbone_name is not in the available models.
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_unfrozen_blocks = num_unfrozen_blocks
        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {self.backbone_name} is not recognized!" 
                             f"Supported backbones are: {self.AVAILABLE_MODELS}")
        
        self.config = self.ARCH_CONFIG[self.backbone_name]
        self.total_layers = self.config['depth']
        self.patch_size = self.config['patch_size']
        
        if num_layers is not None:
            if num_layers > self.total_layers:
                raise ValueError(f"num_layers ({num_layers}) cannot be greater than total layers ({self.total_layers})")
            self.num_layers = num_layers
        else:
            self.num_layers = self.total_layers

        self.print(f"DinoV2 initialized with: ")
        self.print(f"  backbone_name: {self.backbone_name}")
        self.print(f"  num_unfrozen_blocks: {self.num_unfrozen_blocks}")
        self.print(f"  num_layers: {self.num_layers}/{self.total_layers}")
                
        self.dino = torch.hub.load('third_party/dinov2', self.backbone_name, source='local')
        
        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False
        
        # Determine the split point for freezing
        self.split = self.num_layers - self.num_unfrozen_blocks
        if self.split < 0:
            self.split = 0
            
        # unfreeze the last few blocks
        for block in self.dino.blocks[self.split:self.num_layers]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Only apply norm if we ran the full model
        if self.num_layers == self.total_layers:
            for param in self.dino.norm.parameters():
                param.requires_grad = True
        
        self.out_channels = self.dino.embed_dim
        
        # Hook for attention extraction
        self._qkv_out = None
        self.num_register_tokens = self.dino.num_register_tokens
        
        # Register hook on the last active layer
        if self.num_layers > 0:
            target_layer = self.dino.blocks[self.num_layers - 1]
            target_layer.attn.qkv.register_forward_hook(self._hook_qkv)
            
    def _hook_qkv(self, module, input, output):
        self._qkv_out = output
    
    def forward(self, x):
        B, _, H, W = x.shape
        
        # Resize the image to ensure it is divisible by patch_size
        h_new, w_new = (H // self.patch_size) * self.patch_size, (W // self.patch_size) * self.patch_size
        x = CenterCrop((h_new, w_new))(x)

        x = self.dino.prepare_tokens_with_masks(x)
        
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            for blk in self.dino.blocks[:self.split]:
                x = blk(x)
        
        x = x.detach()
        
        # Last blocks are trained
        for blk in self.dino.blocks[self.split:self.num_layers]:
            x = blk(x)
            
        # Only apply norm if we ran the full model
        if self.num_layers == self.total_layers:
            x = self.dino.norm(x)
        
        # ------- Output Handling -------
        # x shape: [B, N_tokens, C]
        # Tokens structure: [CLS, (Registers...), Patches...]
        
        cls_token = x[:, 0]
            
        ph, pw = h_new // self.patch_size, w_new // self.patch_size
        n_patches = ph * pw
        
        # Take the last n_patches tokens
        # Correct handling of register tokens
        # Indices: CLS=0, Regs=1..R, Patches=R+1..
        # Or simply take the last n_patches
        patch_tokens = x[:, -n_patches:, :]
        
        # Reshape to [B, C, H, W]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, self.out_channels, ph, pw)
        
        # ------- Attention Extraction -------
        attn = None
        if self._qkv_out is not None:
            # qkv_out shape: [B, N_tokens, 3*C]
            qkv = self._qkv_out
            self._qkv_out = None # Clear hook output
            
            B_tok, N_tok, _ = qkv.shape
            # Get head info from the last block
            last_blk = self.dino.blocks[self.num_layers - 1]
            num_heads = last_blk.attn.num_heads
            head_dim = self.out_channels // num_heads
            scale = head_dim ** -0.5
            
            qkv = qkv.reshape(B_tok, N_tok, 3, num_heads, head_dim)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2) # [B, H, N, D]
            k = k.transpose(1, 2) # [B, H, N, D]
            
            # Compute attention map
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1) # [B, H, N, N]
            
        return cls_token, patch_tokens, attn
