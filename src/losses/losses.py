"""
losses.py
=========
All training loss functions for the multimodal scattering pipeline.

Training objective
------------------
``L_Total = ω₁·L_BG + ω₂·L_MSE,S + ω₃·(L_MSE,W·G) + ω₄·L_NCE``

Each term is defined as a standalone function for testability and is
composed inside :class:`TotalLoss`.

Loss term descriptions
----------------------

``L_BG`` – Background estimation loss
    Supervises the BackgroundCNN to produce a physically valid background:

    * **Rolling-ball guide** ``‖f_bg − x_ball‖²``:
      Penalises deviation from the rolling-ball morphological estimate.
    * **Smoothness** ``λ₁‖∇²f_bg‖²``:
      Penalises high-frequency oscillations via the discrete second
      derivative (finite difference Laplacian).
    * **Positivity** ``λ₂ mean(ReLU(−f_bg))``:
      Penalises any negative values in the background prediction.

``L_MSE,S`` – SAXS reconstruction loss
    Plain mean-squared error between the decoded SAXS estimate ``x̂_diff,s``
    and the target difference signal ``x_diff,s``.

``L_MSE,W·G`` – WAXS reconstruction loss with prior weighting
    MSE weighted by a spatially non-uniform weight mask *G* that upweights
    specific q-positions (default: positions 50 and 150) where
    biologically relevant WAXS features (e.g. amyloid β-sheet peaks) are
    expected.

``L_NCE`` – InfoNCE contrastive loss
    Encourages the SAXS embedding ``z_S`` and WAXS embedding ``z_W`` from
    the *same* sample to be closer in cosine space than embeddings from
    *different* samples.  Implemented as the symmetric cross-entropy over
    the cosine similarity matrix.

References
----------
- InfoNCE: van den Oord et al., "Representation Learning with Contrastive
  Predictive Coding", arXiv:1807.03748.
- Prior-weighted MSE: domain-specific weighting inspired by peak-sensitive
  loss formulations in spectroscopy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def background_loss(
    f_bg: torch.Tensor,
    x_ball: torch.Tensor,
    lambda1: float = 0.01,
    lambda2: float = 0.1,
) -> torch.Tensor:
    """Compute the three-term background estimation loss ``L_BG``.

    ``L_BG = ‖f_bg − x_ball‖² + λ₁‖∇²f_bg‖² + λ₂ mean(ReLU(−f_bg))``

    Args:
        f_bg:   Predicted background from BackgroundCNN, shape ``(B, L)``.
        x_ball: Rolling-ball guide target,               shape ``(B, L)``.
        lambda1: Weight for the second-derivative smoothness term.
        lambda2: Weight for the positivity penalty.

    Returns:
        Scalar loss tensor.
    """
    # Term 1: MSE guide – pull f_bg towards the rolling-ball estimate
    guide_loss = F.mse_loss(f_bg, x_ball)

    # Term 2: Second-derivative smoothness via discrete Laplacian
    # ∇²f[i] = f[i+1] - 2·f[i] + f[i-1]  (central finite difference)
    second_derivative = f_bg[:, 2:] - 2.0 * f_bg[:, 1:-1] + f_bg[:, :-2]
    smoothness_loss = lambda1 * second_derivative.pow(2).mean()

    # Term 3: Positivity – penalise any negative background values
    positivity_loss = lambda2 * F.relu(-f_bg).mean()

    return guide_loss + smoothness_loss + positivity_loss


def mse_reconstruction_loss(
    x_hat: torch.Tensor,
    x_target: torch.Tensor,
) -> torch.Tensor:
    """Plain mean-squared error reconstruction loss (``L_MSE,S``).

    Args:
        x_hat:    Reconstructed signal from Decoder, shape ``(B, L)``.
        x_target: Ground-truth difference signal,    shape ``(B, L)``.

    Returns:
        Scalar MSE loss tensor.
    """
    return F.mse_loss(x_hat, x_target)


def prior_weighted_mse_loss(
    x_hat_w: torch.Tensor,
    x_target_w: torch.Tensor,
    peak_positions: Sequence[int],
    peak_weight: float = 5.0,
) -> torch.Tensor:
    """Prior-weighted MSE reconstruction loss for WAXS (``L_MSE,W·G``).

    Upweights the reconstruction error at biologically significant q-positions
    (e.g. amyloid β-sheet reflection at indices 50 and 150 in the WAXS array)
    to ensure the model is particularly accurate at those locations.

    The weight mask *G* is a flat vector of ones with ``peak_weight`` at each
    specified position.

    Args:
        x_hat_w:        Reconstructed WAXS difference signal, shape ``(B, L)``.
        x_target_w:     Ground-truth WAXS difference signal,  shape ``(B, L)``.
        peak_positions: Sequence of integer q-indices to upweight.
        peak_weight:    Weight multiplier applied at peak positions.

    Returns:
        Scalar weighted MSE loss tensor.
    """
    signal_len = x_hat_w.shape[-1]

    # Build the per-position weight vector G: default 1.0, peak_weight at peaks
    weight_mask = torch.ones(signal_len, device=x_hat_w.device)
    for pos in peak_positions:
        if 0 <= pos < signal_len:
            weight_mask[pos] = peak_weight

    # Compute element-wise squared errors then apply spatial weighting
    squared_errors = (x_hat_w - x_target_w).pow(2)          # (B, L)
    weighted_loss = (squared_errors * weight_mask.unsqueeze(0)).mean()
    return weighted_loss


def info_nce_loss(
    z_s: torch.Tensor,
    z_w: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss (``L_NCE``).

    Treats SAXS–WAXS pairs from the *same* sample as positives and all
    cross-sample pairs as negatives.  The loss is computed symmetrically
    (SAXS→WAXS and WAXS→SAXS) and averaged.

    The temperature parameter ``τ`` sharpens (small τ) or softens (large τ)
    the similarity distribution.

    Args:
        z_s:         SAXS embeddings,  shape ``(B, D)`` – will be L2-normalised.
        z_w:         WAXS embeddings,  shape ``(B, D)`` – will be L2-normalised.
        temperature: Softmax temperature ``τ`` (default 0.07).

    Returns:
        Scalar InfoNCE loss tensor.

    Note:
        A batch size of at least 2 is required; with B=1 there are no
        negatives and the loss is undefined.
    """
    batch_size = z_s.shape[0]

    # L2-normalise embeddings so dot product equals cosine similarity
    z_s = F.normalize(z_s, dim=-1)
    z_w = F.normalize(z_w, dim=-1)

    # Cosine similarity matrix: (B, B), entry [i, j] = sim(z_s[i], z_w[j])
    logits = torch.mm(z_s, z_w.T) / temperature

    # Positive pairs are on the diagonal (same sample index)
    labels = torch.arange(batch_size, device=z_s.device)

    # Symmetric: average S→W and W→S cross-entropy
    loss_s2w = F.cross_entropy(logits,   labels)
    loss_w2s = F.cross_entropy(logits.T, labels)
    return (loss_s2w + loss_w2s) / 2.0


# ---------------------------------------------------------------------------
# Aggregated loss breakdown
# ---------------------------------------------------------------------------

@dataclass
class LossBreakdown:
    """Named container for the scalar values of each loss component.

    The ``total`` field is the full computational graph tensor used for
    ``.backward()``.  All other fields are detached scalars for logging.

    Attributes:
        total: Weighted sum ``ω₁·L_BG + ω₂·L_MSE,S + ω₃·L_MSE,W·G + ω₄·L_NCE``.
        bg:    Background loss ``L_BG`` (detached).
        mse_s: SAXS reconstruction loss ``L_MSE,S`` (detached).
        mse_w: WAXS prior-weighted reconstruction loss ``L_MSE,W·G`` (detached).
        nce:   InfoNCE contrastive loss ``L_NCE`` (detached).
    """

    total: torch.Tensor
    bg:    torch.Tensor
    mse_s: torch.Tensor
    mse_w: torch.Tensor
    nce:   torch.Tensor


# ---------------------------------------------------------------------------
# Composite loss module
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """Composite training loss module.

    Combines all four loss terms into the total objective:

    ``L_Total = ω₁·L_BG + ω₂·L_MSE,S + ω₃·(L_MSE,W·G) + ω₄·L_NCE``

    Implemented as an ``nn.Module`` so it can be moved to the training
    device alongside the model.

    Args:
        omega1: Weight for ``L_BG`` (background loss).
        omega2: Weight for ``L_MSE,S`` (SAXS reconstruction loss).
        omega3: Weight for ``L_MSE,W·G`` (WAXS prior-weighted reconstruction loss).
        omega4: Weight for ``L_NCE`` (InfoNCE contrastive loss).
        lambda1: Smoothness penalty weight inside ``L_BG``.
        lambda2: Positivity penalty weight inside ``L_BG``.
        waxs_peak_positions: Q-indices to upweight in the WAXS reconstruction loss.
        waxs_peak_weight: Multiplier applied at peak positions.
        temperature: InfoNCE temperature ``τ``.

    Example::

        criterion = TotalLoss()
        breakdown = criterion(
            f_bg_s, f_bg_w, xball_s, xball_w,
            xhat_s, xhat_w, xdiff_s, xdiff_w,
            z_s, z_w
        )
        breakdown.total.backward()
    """

    def __init__(
        self,
        omega1: float = 1.0,
        omega2: float = 1.0,
        omega3: float = 1.0,
        omega4: float = 0.5,
        lambda1: float = 0.01,
        lambda2: float = 0.1,
        waxs_peak_positions: Sequence[int] = (50, 150),
        waxs_peak_weight: float = 5.0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        # Loss weights (omegas)
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        self.omega4 = omega4
        # Background loss regularisation coefficients
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # WAXS prior weighting parameters
        self.peak_positions = list(waxs_peak_positions)
        self.waxs_peak_weight = waxs_peak_weight
        # InfoNCE temperature
        self.temperature = temperature

    def forward(
        self,
        f_bg_s: torch.Tensor,
        f_bg_w: torch.Tensor,
        xball_s: torch.Tensor,
        xball_w: torch.Tensor,
        xhat_s: torch.Tensor,
        xhat_w: torch.Tensor,
        xdiff_s: torch.Tensor,
        xdiff_w: torch.Tensor,
        z_s: torch.Tensor,
        z_w: torch.Tensor,
    ) -> LossBreakdown:
        """Compute all loss components and return their weighted sum.

        Args:
            f_bg_s:  SAXS background prediction, shape ``(B, saxs_dim)``.
            f_bg_w:  WAXS background prediction, shape ``(B, waxs_dim)``.
            xball_s: SAXS rolling-ball guide,     shape ``(B, saxs_dim)``.
            xball_w: WAXS rolling-ball guide,     shape ``(B, waxs_dim)``.
            xhat_s:  Reconstructed SAXS diff,     shape ``(B, saxs_dim)``.
            xhat_w:  Reconstructed WAXS diff,     shape ``(B, waxs_dim)``.
            xdiff_s: Ground-truth SAXS diff,      shape ``(B, saxs_dim)``.
            xdiff_w: Ground-truth WAXS diff,      shape ``(B, waxs_dim)``.
            z_s:     SAXS latent embedding,        shape ``(B, D)``.
            z_w:     WAXS latent embedding,        shape ``(B, D)``.

        Returns:
            :class:`LossBreakdown` with the total (grad-enabled) and
            individual component losses (detached, for logging).
        """
        # L_BG: sum of SAXS and WAXS background losses
        l_bg = (
            background_loss(f_bg_s, xball_s, self.lambda1, self.lambda2)
            + background_loss(f_bg_w, xball_w, self.lambda1, self.lambda2)
        )

        # L_MSE,S: plain SAXS reconstruction
        l_mse_s = mse_reconstruction_loss(xhat_s, xdiff_s)

        # L_MSE,W·G: prior-weighted WAXS reconstruction
        l_mse_w = prior_weighted_mse_loss(
            xhat_w, xdiff_w, self.peak_positions, self.waxs_peak_weight
        )

        # L_NCE: cross-modal contrastive alignment
        l_nce = info_nce_loss(z_s, z_w, self.temperature)

        # Weighted combination
        total = (
            self.omega1 * l_bg
            + self.omega2 * l_mse_s
            + self.omega3 * l_mse_w
            + self.omega4 * l_nce
        )

        return LossBreakdown(
            total=total,
            bg=l_bg.detach(),
            mse_s=l_mse_s.detach(),
            mse_w=l_mse_w.detach(),
            nce=l_nce.detach(),
        )
