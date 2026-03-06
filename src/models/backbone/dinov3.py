import torch
import torch.nn as nn
import os
from torchvision.transforms import CenterCrop
from src.utils.logger import print_rank_0

# RoPE helpers
def rope_rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def rope_apply(x, sin, cos):
    return (x * cos) + (rope_rotate_half(x) * sin)

class DinoV3(nn.Module):
    AVAILABLE_MODELS = [
        'dinov3_vits16',
        'dinov3_vits16plus',
        'dinov3_vitb16',
        'dinov3_vitl16',
        'dinov3_vith16plus',
        'dinov3_vit7b16'
    ]
    
    MODEL_WEIGHTS_MAP = {
        'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
        'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vith16plus': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
        'dinov3_vit7b16': 'dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth',
    }

    ARCH_CONFIG = {
        'dinov3_vits16': {'depth': 12, 'embed_dim': 384, 'patch_size': 16},
        'dinov3_vits16plus': {'depth': 12, 'embed_dim': 384, 'patch_size': 16},
        'dinov3_vitb16': {'depth': 12, 'embed_dim': 768, 'patch_size': 16},
        'dinov3_vitl16': {'depth': 24, 'embed_dim': 1024, 'patch_size': 16},
        'dinov3_vith16plus': {'depth': 32, 'embed_dim': 1280, 'patch_size': 16},
        'dinov3_vit7b16': {'depth': 40, 'embed_dim': 4096, 'patch_size': 16},
    }

    def print(self, msg):
        print_rank_0(f"[{self.__class__.__name__}] {msg}")

    def __init__(
        self,
        backbone_name="dinov3_vitl16",
        num_unfrozen_blocks=0,
        num_layers=None,
    ):
        """DinoV3 backbone with the ability to keep only the last num_unfrozen_blocks trainable.

        Args:
            backbone_name (str, optional): DinoV3 variant. Defaults to "dinov3_vitl16".
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

        self.print(f"DinoV3 initialized with: ")
        self.print(f"  backbone_name: {self.backbone_name}")
        self.print(f"  num_unfrozen_blocks: {self.num_unfrozen_blocks}")
        self.print(f"  num_layers: {self.num_layers}/{self.total_layers}")
        
        # Determine paths
        repo_dir = os.path.join(os.getcwd(), "third_party", "dinov3")
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        weight_filename = self.MODEL_WEIGHTS_MAP.get(self.backbone_name)
        
        if not weight_filename:
             raise ValueError(f"No weight file mapped for {self.backbone_name}")
             
        weights_path = os.path.join(cache_dir, weight_filename)
        
        if not os.path.exists(weights_path):
             # In a real scenario, we might want to download it or warn.
             # For now, we assume the user has placed the weights there or we raise error.
             # Or we can try to load without weights if just testing architecture?
             # But torch.hub.load usually expects weights if we don't say pretrained=False?
             # The previous code raised FileNotFoundError, so I'll keep it.
             raise FileNotFoundError(f"Weights not found at {weights_path}. Please download them first.")

        self.dino = torch.hub.load(repo_dir, self.backbone_name, source='local', weights=weights_path)
        
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
        self.n_storage_tokens = self.dino.n_storage_tokens
        
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

        # DINOv3 returns x and (H_feat, W_feat)
        x, (H_feat, W_feat) = self.dino.prepare_tokens_with_masks(x)
        
        # Prepare RoPE if needed
        rope_sincos = None
        if hasattr(self.dino, 'rope_embed') and self.dino.rope_embed is not None:
            #  print("Preparing RoPE embeddings.")
             rope_sincos = self.dino.rope_embed(H=H_feat, W=W_feat)
        
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            for i, blk in enumerate(self.dino.blocks[:self.split]):
                x = blk(x, rope_sincos)
        
        x = x.detach()
        
        # Last blocks are trained
        for i, blk in enumerate(self.dino.blocks[self.split:self.num_layers]):
            x = blk(x, rope_sincos)
            
        # Only apply norm if we ran the full model
        if self.num_layers == self.total_layers:
            # Check if we need to apply different norms for CLS/Registers and Patches
            if self.dino.untie_cls_and_patch_norms:
                # print("Applying separate norms for CLS/Storage and Patch tokens.")
                # x structure: [CLS, Storage..., Patches...]
                # Split tokens
                n_special = 1 + self.n_storage_tokens
                
                x_special = x[:, :n_special]
                x_patches = x[:, n_special:]
                
                # Apply respective norms
                x_special = self.dino.cls_norm(x_special)
                x_patches = self.dino.norm(x_patches)
                
                # Recombine
                x = torch.cat([x_special, x_patches], dim=1)
            else:
                # Standard case: one norm for everything
                x = self.dino.norm(x)
        
        # ------- Output Handling -------
        # x shape: [B, N_tokens, C]
        # Tokens: [CLS, Storage..., Patches...]
        
        cls_token = x[:, 0]
            
        # Calculate number of patches
        n_patches = H_feat * W_feat
        
        # Take the last n_patches tokens
        patch_tokens = x[:, -n_patches:, :]
        
        # Reshape to [B, C, H, W]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, self.out_channels, H_feat, W_feat)
        
        # ------- Attention Extraction -------
        attn = None
        if self._qkv_out is not None:
            qkv = self._qkv_out
            self._qkv_out = None
            
            B_tok, N_tok, _ = qkv.shape
            last_blk = self.dino.blocks[self.num_layers - 1]
            num_heads = last_blk.attn.num_heads
            head_dim = self.out_channels // num_heads
            scale = head_dim ** -0.5
            
            qkv = qkv.reshape(B_tok, N_tok, 3, num_heads, head_dim)
            q, k, v = qkv.unbind(2)
            q = q.transpose(1, 2) # [B, H, N, D]
            k = k.transpose(1, 2) # [B, H, N, D]
            
            # Apply RoPE
            if rope_sincos is not None:
                # rope_sincos is (sin, cos)
                # DINOv3 SelfAttentionBlock._maybe_index_rope handles batch dim if needed
                # But here rope_sincos is likely [H_feat, W_feat, D] or similar?
                # Wait, rope_embed(H, W) returns (sin, cos).
                # sin/cos shape: [H*W, D/heads] or similar?
                # Let's check SelfAttentionBlock._maybe_index_rope logic
                # It handles if sin.ndim == 4 (batch) or not.
                # Here we assume standard inference, rope_sincos is likely [N_patches, D_head]
                # But wait, tokens include CLS and Storage. RoPE usually only applies to patches?
                # Or it applies to all but with offset?
                # DINOv3 RoPE implementation:
                # q_prefix = q[:, :, :prefix, :]
                # q = rope_apply(q[:, :, prefix:, :], sin, cos)
                # prefix = N - sin.shape[-2]
                
                sin, cos = rope_sincos
                # sin, cos are likely on same device as parameters, but x is on device.
                # Ensure device match
                sin = sin.to(q.device)
                cos = cos.to(q.device)
                
                # Check shapes
                # q: [B, H, N_total, D_head]
                # sin: [N_patches, D_head] (usually)
                
                prefix = N_tok - sin.shape[0] # sin.shape[-2] if it has head dim?
                # Usually RoPE is [seq_len, dim]
                
                if prefix >= 0:
                    q_patches = q[:, :, prefix:, :]
                    k_patches = k[:, :, prefix:, :]
                    
                    # Apply RoPE
                    # Need to broadcast sin/cos to [B, H, N_patches, D_head]
                    # sin is [N_patches, D_head] -> [1, 1, N_patches, D_head]
                    
                    q_patches = rope_apply(q_patches, sin, cos)
                    k_patches = rope_apply(k_patches, sin, cos)
                    
                    q = torch.cat((q[:, :, :prefix, :], q_patches), dim=2)
                    k = torch.cat((k[:, :, :prefix, :], k_patches), dim=2)
            
            # Compute attention
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            
        return cls_token, patch_tokens, attn
