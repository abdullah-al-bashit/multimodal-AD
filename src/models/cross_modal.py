"""Cross-modal fusion: Z_mp = fuse(z_S, z_W)."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class CrossModalFusion(nn.Module):
    """
    Methods: "attention" (cross-MHA) | "concat" (MLP) | "add" (element-wise).
    """
    def __init__(self, latent_dim=64, method="attention", out_dim=None):
        super().__init__()
        self.method = method
        out_dim = out_dim or latent_dim
        if method == "attention":
            self.attn_s2w = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
            self.attn_w2s = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
            self.proj = nn.Sequential(nn.Linear(latent_dim*2, out_dim),
                                      nn.LayerNorm(out_dim), nn.GELU())
        elif method == "concat":
            self.proj = nn.Sequential(nn.Linear(latent_dim*2, out_dim),
                                      nn.LayerNorm(out_dim), nn.GELU())
        elif method == "add":
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion method: '{method}'")

    def forward(self, z_s, z_w):
        if self.method == "attention":
            qs, qw = z_s.unsqueeze(1), z_w.unsqueeze(1)
            cs, _  = self.attn_s2w(qs, qw, qw)
            cw, _  = self.attn_w2s(qw, qs, qs)
            return self.proj(torch.cat([cs.squeeze(1), cw.squeeze(1)], dim=-1))
        if self.method == "concat":
            return self.proj(torch.cat([z_s, z_w], dim=-1))
        return self.proj(z_s + z_w)
