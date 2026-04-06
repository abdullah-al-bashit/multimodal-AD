"""
scattering_model.py
===================
Top-level end-to-end multimodal scattering model.

Architecture summary (see diagram in docs/architecture.pdf)
-----------------------------------------------------------
::

    Input x ∈ ℝ^590
         │
      split()
    ┌────┴────┐
    xs(300)  xw(290)          ← SAXS / WAXS signals
    │  │      │  │
    │ morph   │ morph         ← rolling-ball BG (precomputed, passed in)
    │  │      │  │
    │ BackgroundCNN           ← f_bg_s, f_bg_w  [supervised by L_BG]
    │  │      │  │
    xs−f_bg  xw−f_bg          ← x_diff_s, x_diff_w
       │         │
    GNN_S(k-NN) GNN_W(k-NN)   ← z_S, z_W  ∈ ℝ^(B×D)
       │         │
       └── fuse ─┘             ← Z_mp (cross-modal, supervised by L_NCE)
       │         │
   Decoder_S  Decoder_W       ← x̂_diff_s, x̂_diff_w  [supervised by L_MSE]

Training objective
------------------
``L_Total = ω₁·L_BG + ω₂·L_MSE,S + ω₃·(L_MSE,W·G) + ω₄·L_NCE``

All loss terms are computed externally in :mod:`src.losses.losses`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import Config
from .background_cnn import BackgroundCNN
from .cross_modal import CrossModalFusion
from .decoder import Decoder
from .gnn_encoder import GNNEncoder


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ModelOutput:
    """Named container for all tensors produced by a single forward pass.

    Using a dataclass instead of a plain tuple makes downstream code
    (trainer, loss functions) more readable and less error-prone.

    Attributes:
        f_bg_s:  SAXS background estimate from BackgroundCNN, shape ``(B, 300)``.
        f_bg_w:  WAXS background estimate from BackgroundCNN, shape ``(B, 290)``.
        xdiff_s: SAXS difference signal ``xs − f_bg_s``,       shape ``(B, 300)``.
        xdiff_w: WAXS difference signal ``xw − f_bg_w``,       shape ``(B, 290)``.
        z_s:     SAXS latent embedding from GNN_S,              shape ``(B, D)``.
        z_w:     WAXS latent embedding from GNN_W,              shape ``(B, D)``.
        z_mp:    Fused cross-modal embedding Z_mp,              shape ``(B, D)``.
        xhat_s:  Reconstructed SAXS difference signal,          shape ``(B, 300)``.
        xhat_w:  Reconstructed WAXS difference signal,          shape ``(B, 290)``.
    """

    f_bg_s:  torch.Tensor  # (B, saxs_dim)
    f_bg_w:  torch.Tensor  # (B, waxs_dim)
    xdiff_s: torch.Tensor  # (B, saxs_dim)
    xdiff_w: torch.Tensor  # (B, waxs_dim)
    z_s:     torch.Tensor  # (B, D)
    z_w:     torch.Tensor  # (B, D)
    z_mp:    torch.Tensor  # (B, D)
    xhat_s:  torch.Tensor  # (B, saxs_dim)
    xhat_w:  torch.Tensor  # (B, waxs_dim)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ScatteringModel(nn.Module):
    """End-to-end multimodal SAXS/WAXS scattering model.

    Orchestrates four sub-modules:

    1. **BackgroundCNN** (×2, one per modality) – estimates the smooth
       scattering background ``f_bg`` guided by the rolling-ball estimate.
    2. **GNNEncoder** (×2) – encodes the background-subtracted difference
       signal as a k-NN graph and produces a fixed-size embedding.
    3. **CrossModalFusion** (×1) – fuses SAXS and WAXS embeddings into
       a shared multimodal representation.
    4. **Decoder** (×2, one per modality) – reconstructs the difference
       signal from the per-modality embeddings.

    Args:
        saxs_dim: SAXS signal length (default 300).
        waxs_dim: WAXS signal length (default 290).
        bg_hidden: Hidden channel widths for the BackgroundCNN body.
        bg_ks: Kernel size for BackgroundCNN convolutions.
        bg_dp: Dropout rate inside BackgroundCNN residual blocks.
        gnn_hidden: Hidden width of intermediate GATv2Conv layers.
        gnn_latent: Latent embedding dimensionality *D*.
        gnn_layers: Number of GATv2Conv layers.
        gnn_k: k for k-NN graph construction.
        gnn_heads: Number of attention heads (non-final layers).
        gnn_dp: Dropout rate in GNNEncoder.
        fusion: CrossModalFusion method – ``"attention"``, ``"concat"``,
            or ``"add"``.
        dec_hidden: Hidden width for the Decoder MLP.
        dec_dp: Dropout rate inside the Decoder.

    Example::

        model = ScatteringModel()
        out = model(xs, xw, xball_s, xball_w)
        print(out.z_mp.shape)   # (B, 64)
    """

    def __init__(
        self,
        saxs_dim: int = 300,
        waxs_dim: int = 290,
        bg_hidden: tuple = (64, 128, 64),
        bg_ks: int = 5,
        bg_dp: float = 0.1,
        gnn_hidden: int = 128,
        gnn_latent: int = 64,
        gnn_layers: int = 3,
        gnn_k: int = 8,
        gnn_heads: int = 4,
        gnn_dp: float = 0.1,
        fusion: str = "attention",
        dec_hidden: int = 128,
        dec_dp: float = 0.1,
    ) -> None:
        super().__init__()
        self.saxs_dim = saxs_dim
        self.waxs_dim = waxs_dim

        # -- Background CNNs (one per modality) ----------------------------
        self.bg_s = BackgroundCNN(saxs_dim, list(bg_hidden), bg_ks, bg_dp)
        self.bg_w = BackgroundCNN(waxs_dim, list(bg_hidden), bg_ks, bg_dp)

        # -- GNN encoders (one per modality) --------------------------------
        self.gnn_s = GNNEncoder(
            signal_len=saxs_dim,
            hidden_dim=gnn_hidden,
            latent_dim=gnn_latent,
            num_layers=gnn_layers,
            k_neighbors=gnn_k,
            heads=gnn_heads,
            dropout=gnn_dp,
        )
        self.gnn_w = GNNEncoder(
            signal_len=waxs_dim,
            hidden_dim=gnn_hidden,
            latent_dim=gnn_latent,
            num_layers=gnn_layers,
            k_neighbors=gnn_k,
            heads=gnn_heads,
            dropout=gnn_dp,
        )

        # -- Cross-modal fusion --------------------------------------------
        self.fusion = CrossModalFusion(gnn_latent, fusion, gnn_latent)

        # -- Decoders (one per modality) -----------------------------------
        self.decoder_s = Decoder(gnn_latent, saxs_dim, dec_hidden, dec_dp)
        self.decoder_w = Decoder(gnn_latent, waxs_dim, dec_hidden, dec_dp)

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Config) -> "ScatteringModel":
        """Instantiate a :class:`ScatteringModel` from a :class:`~src.config.Config`.

        This is the recommended construction path in training scripts as it
        keeps hyper-parameters in one place.

        Args:
            cfg: Fully-populated configuration object.

        Returns:
            A new :class:`ScatteringModel` instance.
        """
        c = cfg
        return cls(
            saxs_dim=c.data.saxs_dim,
            waxs_dim=c.data.waxs_dim,
            bg_hidden=c.background_cnn.hidden_channels,
            bg_ks=c.background_cnn.kernel_size,
            bg_dp=c.background_cnn.dropout,
            gnn_hidden=c.gnn.hidden_dim,
            gnn_latent=c.gnn.latent_dim,
            gnn_layers=c.gnn.num_layers,
            gnn_k=c.gnn.k_neighbors,
            gnn_heads=c.gnn.heads,
            gnn_dp=c.gnn.dropout,
            fusion=c.cross_modal.method,
            dec_hidden=c.decoder.hidden_dim,
            dec_dp=c.decoder.dropout,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        xs: torch.Tensor,
        xw: torch.Tensor,
        xball_s: torch.Tensor,
        xball_w: torch.Tensor,
    ) -> ModelOutput:
        """Full forward pass producing all intermediate and final tensors.

        Steps:
          1. Estimate backgrounds ``f_bg_s``, ``f_bg_w`` via BackgroundCNN.
          2. Subtract backgrounds to get difference signals.
          3. Encode each difference signal via the corresponding GNN.
          4. Fuse embeddings cross-modally.
          5. Decode each embedding back to a difference-signal estimate.

        Args:
            xs:      Normalised SAXS signal,         shape ``(B, saxs_dim)``.
            xw:      Normalised WAXS signal,         shape ``(B, waxs_dim)``.
            xball_s: SAXS rolling-ball BG guide,     shape ``(B, saxs_dim)``.
            xball_w: WAXS rolling-ball BG guide,     shape ``(B, waxs_dim)``.

        Returns:
            :class:`ModelOutput` dataclass containing all intermediate and
            output tensors needed to compute the training losses.
        """
        # Step 1: background estimation (supervised by L_BG)
        f_bg_s = self.bg_s(xs, xball_s)   # (B, saxs_dim)
        f_bg_w = self.bg_w(xw, xball_w)   # (B, waxs_dim)

        # Step 2: background subtraction → difference signals
        xdiff_s = xs - f_bg_s   # (B, saxs_dim)
        xdiff_w = xw - f_bg_w   # (B, waxs_dim)

        # Step 3: per-modality GNN encoding
        z_s = self.gnn_s(xdiff_s)   # (B, D)
        z_w = self.gnn_w(xdiff_w)   # (B, D)

        # Step 4: cross-modal fusion (supervised by L_NCE)
        z_mp = self.fusion(z_s, z_w)   # (B, D)

        # Step 5: per-modality decoding (supervised by L_MSE_S / L_MSE_W·G)
        xhat_s = self.decoder_s(z_s)   # (B, saxs_dim)
        xhat_w = self.decoder_w(z_w)   # (B, waxs_dim)

        return ModelOutput(
            f_bg_s=f_bg_s,
            f_bg_w=f_bg_w,
            xdiff_s=xdiff_s,
            xdiff_w=xdiff_w,
            z_s=z_s,
            z_w=z_w,
            z_mp=z_mp,
            xhat_s=xhat_s,
            xhat_w=xhat_w,
        )

    def encode(
        self,
        xs: torch.Tensor,
        xw: torch.Tensor,
        xball_s: torch.Tensor,
        xball_w: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method: run forward pass and return only ``Z_mp``.

        Useful for inference tasks (e.g. clustering, classification) where
        only the fused embedding is needed.

        Args:
            xs:      Normalised SAXS signal,       shape ``(B, saxs_dim)``.
            xw:      Normalised WAXS signal,       shape ``(B, waxs_dim)``.
            xball_s: SAXS rolling-ball BG guide,   shape ``(B, saxs_dim)``.
            xball_w: WAXS rolling-ball BG guide,   shape ``(B, waxs_dim)``.

        Returns:
            Fused embedding ``Z_mp`` of shape ``(B, D)``.
        """
        return self.forward(xs, xw, xball_s, xball_w).z_mp
