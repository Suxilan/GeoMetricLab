import torch
import torch.nn as nn
class SoftP(nn.Module):
    """
    Soft Probing (SoftP) from SAGE:
      s_i = ||X_i||_2 + eps
      beta_i = alpha * sigmoid(phi(s_i))
      X̃_i = (1 + beta_i) * X_i
    """
    def __init__(self, alpha: float = 1.0, hidden: int = 32, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        # phi: 2-layer MLP on scalar s_i (shape [..., 1])
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(1) * alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, M) patch descriptors (exclude cls token)
        returns: (B, L, M) modulated descriptors
        """
        # s_i = ||X_i||_2 + eps
        x_norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True) + self.eps  # (B, L, 1)
        beta = self.alpha * self.mlp(x_norm)                                   # (B, L, 1)
        return x * (1.0 + beta)    