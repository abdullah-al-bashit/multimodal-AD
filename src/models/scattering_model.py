"""Full end-to-end multi-modal scattering model."""
from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn as nn
from .background_cnn    import BackgroundCNN
from .gnn_encoder       import GNNEncoder
from .cross_modal       import CrossModalFusion
from .decoder           import Decoder
from ..config           import Config

@dataclass
class ModelOutput:
    f_bg_s:  torch.Tensor   # (B, 300)
    f_bg_w:  torch.Tensor   # (B, 290)
    xdiff_s: torch.Tensor   # (B, 300)
    xdiff_w: torch.Tensor   # (B, 290)
    z_s:     torch.Tensor   # (B, D)
    z_w:     torch.Tensor   # (B, D)
    z_mp:    torch.Tensor   # (B, D)
    xhat_s:  torch.Tensor   # (B, 300)
    xhat_w:  torch.Tensor   # (B, 290)

class ScatteringModel(nn.Module):
    def __init__(self, saxs_dim=300, waxs_dim=290,
                 bg_hidden=(64,128,64), bg_ks=5, bg_dp=0.1,
                 gnn_hidden=128, gnn_latent=64, gnn_layers=3,
                 gnn_k=8, gnn_heads=4, gnn_dp=0.1,
                 fusion="attention", dec_hidden=128, dec_dp=0.1):
        super().__init__()
        self.saxs_dim = saxs_dim
        self.waxs_dim = waxs_dim
        self.bg_s = BackgroundCNN(saxs_dim, list(bg_hidden), bg_ks, bg_dp)
        self.bg_w = BackgroundCNN(waxs_dim, list(bg_hidden), bg_ks, bg_dp)
        self.gnn_s = GNNEncoder(saxs_dim, hidden_dim=gnn_hidden, latent_dim=gnn_latent,
                                num_layers=gnn_layers, k_neighbors=gnn_k,
                                heads=gnn_heads, dropout=gnn_dp)
        self.gnn_w = GNNEncoder(waxs_dim, hidden_dim=gnn_hidden, latent_dim=gnn_latent,
                                num_layers=gnn_layers, k_neighbors=gnn_k,
                                heads=gnn_heads, dropout=gnn_dp)
        self.fusion   = CrossModalFusion(gnn_latent, fusion, gnn_latent)
        self.decoder_s = Decoder(gnn_latent, saxs_dim, dec_hidden, dec_dp)
        self.decoder_w = Decoder(gnn_latent, waxs_dim, dec_hidden, dec_dp)

    @classmethod
    def from_config(cls, cfg: Config) -> "ScatteringModel":
        c = cfg
        return cls(
            saxs_dim=c.data.saxs_dim,       waxs_dim=c.data.waxs_dim,
            bg_hidden=c.background_cnn.hidden_channels,
            bg_ks=c.background_cnn.kernel_size, bg_dp=c.background_cnn.dropout,
            gnn_hidden=c.gnn.hidden_dim,    gnn_latent=c.gnn.latent_dim,
            gnn_layers=c.gnn.num_layers,    gnn_k=c.gnn.k_neighbors,
            gnn_heads=c.gnn.heads,          gnn_dp=c.gnn.dropout,
            fusion=c.cross_modal.method,
            dec_hidden=c.decoder.hidden_dim, dec_dp=c.decoder.dropout)

    def forward(self, xs, xw, xball_s, xball_w) -> ModelOutput:
        f_bg_s  = self.bg_s(xs, xball_s)
        f_bg_w  = self.bg_w(xw, xball_w)
        xdiff_s = xs - f_bg_s
        xdiff_w = xw - f_bg_w
        z_s     = self.gnn_s(xdiff_s)
        z_w     = self.gnn_w(xdiff_w)
        z_mp    = self.fusion(z_s, z_w)
        return ModelOutput(f_bg_s, f_bg_w, xdiff_s, xdiff_w, z_s, z_w, z_mp,
                           self.decoder_s(z_s), self.decoder_w(z_w))

    def encode(self, xs, xw, xball_s, xball_w) -> torch.Tensor:
        return self.forward(xs, xw, xball_s, xball_w).z_mp
