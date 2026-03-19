"""MLP decoder with residual skip and Softplus output (non-negative)."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class _Block(nn.Module):
    def __init__(self, i, o, dp): super().__init__(); self.net = nn.Sequential(
        nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(), nn.Dropout(dp))
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=300, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.b1   = _Block(latent_dim,   hidden_dim,   dropout)
        self.b2   = _Block(hidden_dim,   hidden_dim*2, dropout)
        self.skip = nn.Linear(latent_dim, hidden_dim*2, bias=False)
        self.out  = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, z):
        h = self.b2(self.b1(z)) + self.skip(z)
        return F.softplus(self.out(h))
