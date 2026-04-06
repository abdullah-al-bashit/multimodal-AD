"""
background_cnn.py
=================
1-D convolutional neural network for background estimation.

Architecture
------------
The network takes two aligned 1-D signals as input:

* **x_signal** – the raw (or normalised) scattering signal ``x_s`` or ``x_w``.
* **x_ball**   – the rolling-ball background estimate computed during
  preprocessing (serves as a physics-informed guide channel).

Both channels are stacked along the channel axis → shape ``(B, 2, L)``,
then processed by:

    Stem (2→C₀) → Body (stacked Conv+BN+ReLU+ResBlock) → Head (1×1 conv)
                                                         → ReLU (non-negative)

The non-negative ReLU ensures the predicted background ``f_bg`` obeys the
physical constraint that scattering intensities cannot be negative.

Role in the full pipeline
--------------------------
``f_bg = BackgroundCNN(x_signal, x_ball)``

The difference signal ``x_diff = x_signal − f_bg`` is then fed to the
downstream GNN encoder.  The quality of ``f_bg`` is supervised through
``L_BG`` which penalises:

1. Deviation from the rolling-ball guide  (MSE term)
2. Non-smoothness of the background       (second-derivative term)
3. Negativity                             (ReLU positivity term)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """1-D residual block: ``Conv → BN → ReLU → Dropout → Conv → BN → (+skip) → ReLU``.

    Keeps the spatial dimension and channel count identical so that the
    skip connection requires no projection layer.

    Args:
        channels: Number of input *and* output channels.
        kernel_size: Convolutional kernel size (should be odd for symmetric
            padding to preserve the sequence length).
        dropout: Dropout probability applied between the two convolutions.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Same-size padding keeps sequence length constant
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with identity skip connection.

        Args:
            x: Feature map of shape ``(B, C, L)``.

        Returns:
            Output feature map of shape ``(B, C, L)``.
        """
        # Pre-activation residual: apply ReLU *after* adding the skip
        return F.relu(self.net(x) + x, inplace=True)


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class BackgroundCNN(nn.Module):
    """Physics-guided 1-D CNN background estimator.

    Predicts the smooth scattering background ``f_bg`` for a single
    modality (SAXS *or* WAXS).  Two separate ``BackgroundCNN`` instances
    are used inside ``ScatteringModel`` – one per modality.

    Network structure
    -----------------
    ::

        [x_signal, x_ball]          # stacked → (B, 2, L)
            │
        Stem: Conv1d(2 → C₀)        # initial feature projection
            │
        Body: for each Cᵢ in hidden_channels[1:]:
               Conv1d(Cᵢ₋₁ → Cᵢ) + BN + ReLU
               ResBlock1d(Cᵢ)
            │
        Head: Conv1d(Cₙ → 1, kernel=1)  # point-wise projection to scalar
            │
        ReLU                             # enforce non-negativity of f_bg

    Args:
        signal_len: Number of data points in the signal (300 for SAXS,
            290 for WAXS).  Not used in the conv computation itself but
            kept for documentation clarity.
        hidden_channels: List of channel widths for stem + body, e.g.
            ``[64, 128, 64]``.  Must have at least one element.
        kernel_size: Shared kernel size for all convolutions (odd integer).
        dropout: Dropout rate inside each :class:`ResBlock1d`.

    Example::

        bg_cnn = BackgroundCNN(signal_len=300, hidden_channels=[64, 128, 64])
        f_bg = bg_cnn(xs, xball_s)   # f_bg: (B, 300)
    """

    def __init__(
        self,
        signal_len: int,
        hidden_channels: List[int] = (64, 128, 64),  # type: ignore[assignment]
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        # -- Stem: fuse the two input channels into the first hidden width ---
        self.stem = nn.Sequential(
            nn.Conv1d(2, hidden_channels[0], kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(inplace=True),
        )

        # -- Body: progressively widen / narrow the feature maps -------------
        body_layers: List[nn.Module] = []
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            body_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                ResBlock1d(out_ch, kernel_size, dropout),
            ])
            in_ch = out_ch
        self.body = nn.Sequential(*body_layers)

        # -- Head: 1×1 conv collapses channels to a single output channel ----
        self.head = nn.Conv1d(in_ch, 1, kernel_size=1)

    def forward(self, x_signal: torch.Tensor, x_ball: torch.Tensor) -> torch.Tensor:
        """Estimate the smooth background given a signal and its rolling-ball guide.

        Args:
            x_signal: Raw (or normalised) scattering signal of shape ``(B, L)``.
            x_ball:   Rolling-ball background guide of shape ``(B, L)``.
                      Must have the same spatial length *L* as *x_signal*.

        Returns:
            Estimated background ``f_bg`` of shape ``(B, L)``, guaranteed
            non-negative by the final ``F.relu``.
        """
        # Stack along channel dim: (B, L) + (B, L) → (B, 2, L)
        h = torch.stack([x_signal, x_ball], dim=1)
        h = self.stem(h)
        h = self.body(h)
        # Head: (B, 1, L) → squeeze → (B, L)
        f_bg = F.relu(self.head(h).squeeze(1), inplace=True)
        return f_bg
