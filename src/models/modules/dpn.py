import torch
import torch.nn as nn

class DPN(nn.Module):
    def __init__(self, D: int, d: int = 32, eps: float = 1e-6):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(D, d),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.activation = nn.Sigmoid()
        self.eps = eps

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        # G: [B, L, D]
        B, L, D = G.shape

        avg_G = self.avg_pool(G.transpose(-1, -2)).squeeze(-1)  # [B, D]
        proj = self.projection(avg_G)  # [B, 1]
        p = self.activation(proj).unsqueeze(-1)  # [B, 1, 1]

        sign = torch.sign(G)
        pow_G = torch.pow(torch.abs(G) + self.eps, p.expand(-1, L, D))
        return sign * pow_G + G