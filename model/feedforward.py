"""
model/feedforward.py
────────────────────
T5 uses:
  - RMSNorm (no bias, no mean subtraction) instead of LayerNorm
  - ReLU feed-forward with a separate gate projection (DenseReluDense)
  - No bias in any linear layer (following T5 v1.1 / T5-base)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.scale).to(x.dtype)


class DenseReluDense(nn.Module):
    """
    Standard T5 FFN:
        FFN(x) = dropout( relu( x W_in ) ) W_out
    Both projections are bias-free.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.wi      = nn.Linear(d_model, d_ff,    bias=False)
        self.wo      = nn.Linear(d_ff,    d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.dropout(F.relu(self.wi(x))))


class TransformerFFN(nn.Module):
    """Pre-norm wrapper: RMSNorm → FFN → residual."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm    = RMSNorm(d_model)
        self.ffn     = DenseReluDense(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ffn(self.norm(x)))
