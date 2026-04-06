"""
decoder.py
==========
MLP decoder with a residual skip connection and Softplus output activation.

Purpose
-------
The decoder reconstructs the background-subtracted difference signal
``x̂_diff`` from a latent embedding ``z ∈ ℝ^D`` produced by the GNN encoder.
Reconstruction quality is measured by ``L_MSE`` (and the prior-weighted
variant ``L_MSE,W·G`` for the WAXS decoder), which drive the encoder to
learn a compact yet information-preserving representation.

Architecture
------------
::

    z ∈ ℝ^D
      │
      ├── skip: Linear(D → 2H)            # direct residual path
      │
      ├── b1: Linear(D→H)  + LN + GELU + Dropout
      │
      └── b2: Linear(H→2H) + LN + GELU + Dropout
            │
          (b2 output) + skip              # residual merge
            │
          Linear(2H → output_dim)
            │
          Softplus                        # ensures non-negative reconstruction

The Softplus activation (a smooth approximation to ReLU) enforces that the
reconstructed difference signal ``x̂_diff`` is non-negative, which is
physically consistent for scattering intensities.

Two decoder instances are used in the full model:
* **Decoder_S** – reconstructs SAXS difference signal (output_dim=300)
* **Decoder_W** – reconstructs WAXS difference signal (output_dim=290)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class _MlpBlock(nn.Module):
    """A single fully-connected block: ``Linear → LayerNorm → GELU → Dropout``.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        dropout: Dropout probability.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """MLP decoder with a residual skip connection for signal reconstruction.

    Args:
        latent_dim: Dimensionality *D* of the input embedding ``z``.
        output_dim: Number of output points (300 for SAXS, 290 for WAXS).
        hidden_dim: Width *H* of the first hidden layer.  The second hidden
            layer uses width ``2H`` to gradually expand towards the output
            dimension.
        dropout: Dropout probability in each :class:`_MlpBlock`.

    Example::

        decoder = Decoder(latent_dim=64, output_dim=300)
        x_hat = decoder(z_s)   # z_s: (B, 64)  →  x_hat: (B, 300)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        output_dim: int = 300,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Two-stage expansion: D → H → 2H
        self.block1 = _MlpBlock(latent_dim, hidden_dim, dropout)
        self.block2 = _MlpBlock(hidden_dim, hidden_dim * 2, dropout)

        # Residual skip from input directly to the pre-output width (2H)
        # Bias=False since LayerNorm in block2 already handles centering
        self.skip = nn.Linear(latent_dim, hidden_dim * 2, bias=False)

        # Final projection to output signal length
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent embedding into a reconstructed difference signal.

        Args:
            z: Latent embedding of shape ``(B, latent_dim)``.

        Returns:
            Reconstructed difference signal ``x̂_diff`` of shape
            ``(B, output_dim)``, non-negative due to Softplus activation.
        """
        # Main path: z → block1 → block2
        h = self.block2(self.block1(z))

        # Add residual skip to preserve gradient flow and reduce vanishing
        h = h + self.skip(z)

        # Project to output dimension and apply Softplus for non-negativity
        # Softplus: log(1 + exp(x)) – smooth, differentiable alternative to ReLU
        return F.softplus(self.output_proj(h))
