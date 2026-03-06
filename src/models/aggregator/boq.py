# Code adapted from amaralibey/Bag-of-Queries: https://github.com/amaralibey/Bag-of-Queries

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any, Dict
from .base import AggregatorBase

class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()
        
        self.encoder = nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4*in_dim, batch_first=True, dropout=0.)
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_dim))
        
        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_dim)
        #####
        
        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(in_dim)
        

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)
        
        q = self.queries.repeat(B, 1, 1)
        
        # the following two lines are used during training.
        # for stability purposes 
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        #######
        
        out, attn = self.cross_attn(q, x, x)        
        out = self.norm_out(out)
        return x, out, attn.detach()

class BoQ(AggregatorBase):
    def __init__(self, in_channels=1024, proj_channels=512, num_queries=32, num_layers=2, row_dim=32):
        super().__init__()
        self.proj_c = nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = nn.LayerNorm(proj_channels)
        
        in_dim = proj_channels
        self.boqs = nn.ModuleList([
            BoQBlock(in_dim, num_queries, nheads=in_dim//64) for _ in range(num_layers)])
        
        self.fc = nn.Linear(num_layers*num_queries, row_dim)

        self.out_channels = in_dim * row_dim
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # reduce input dimension using 3x3 conv when using ResNet
        x = self.proj_c(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        
        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1))
        out = out.flatten(1)
        return out, {'attns': attns}