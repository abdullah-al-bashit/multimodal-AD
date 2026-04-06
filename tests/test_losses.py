"""
test_losses.py
==============
Unit tests for :mod:`src.losses.losses`.

Covers:
  * :func:`~src.losses.losses.background_loss` – positivity and term contributions.
  * :func:`~src.losses.losses.info_nce_loss` – scalar output and temperature scaling.
  * :func:`~src.losses.losses.prior_weighted_mse_loss` – peak weighting effect.
  * :class:`~src.losses.losses.TotalLoss` – forward pass produces non-negative
    total and correct LossBreakdown fields.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Allow running tests directly from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses.losses import (
    TotalLoss,
    background_loss,
    info_nce_loss,
    prior_weighted_mse_loss,
)

# Shared constants
BATCH_SIZE  = 4
LATENT_DIM  = 64
SAXS_DIM    = 300
WAXS_DIM    = 290


# ---------------------------------------------------------------------------
# background_loss
# ---------------------------------------------------------------------------

class TestBackgroundLoss:
    """Tests for the three-term background estimation loss L_BG."""

    def test_loss_is_non_negative(self):
        """L_BG must be ≥ 0 for any non-negative f_bg and x_ball inputs."""
        f_bg   = torch.abs(torch.randn(BATCH_SIZE, SAXS_DIM))
        x_ball = torch.abs(torch.randn(BATCH_SIZE, SAXS_DIM))
        loss = background_loss(f_bg, x_ball)
        assert loss.item() >= 0, f"Background loss was negative: {loss.item()}"

    def test_positivity_penalty_fires_on_negative_values(self):
        """L_BG must increase when f_bg contains negative values."""
        x_ball = torch.ones(BATCH_SIZE, SAXS_DIM)
        f_bg_pos = torch.ones(BATCH_SIZE, SAXS_DIM)            # all positive
        f_bg_neg = -torch.ones(BATCH_SIZE, SAXS_DIM)           # all negative

        loss_pos = background_loss(f_bg_pos, x_ball, lambda2=1.0)
        loss_neg = background_loss(f_bg_neg, x_ball, lambda2=1.0)
        assert loss_neg > loss_pos, "Positivity penalty did not increase loss for negative f_bg."


# ---------------------------------------------------------------------------
# info_nce_loss
# ---------------------------------------------------------------------------

class TestInfoNCELoss:
    """Tests for the symmetric InfoNCE contrastive loss L_NCE."""

    def test_output_is_scalar(self):
        """info_nce_loss must return a zero-dimensional tensor."""
        loss = info_nce_loss(torch.randn(BATCH_SIZE, LATENT_DIM), torch.randn(BATCH_SIZE, LATENT_DIM))
        assert loss.dim() == 0, f"Expected scalar, got dim={loss.dim()}"

    def test_loss_is_non_negative(self):
        """InfoNCE is a cross-entropy loss and must always be ≥ 0."""
        loss = info_nce_loss(torch.randn(BATCH_SIZE, LATENT_DIM), torch.randn(BATCH_SIZE, LATENT_DIM))
        assert loss.item() >= 0, f"InfoNCE loss was negative: {loss.item()}"

    def test_lower_temperature_increases_loss(self):
        """Sharper temperature should generally produce higher (harder) loss."""
        z_s = torch.randn(BATCH_SIZE, LATENT_DIM)
        z_w = torch.randn(BATCH_SIZE, LATENT_DIM)
        loss_low_temp  = info_nce_loss(z_s, z_w, temperature=0.01)
        loss_high_temp = info_nce_loss(z_s, z_w, temperature=1.0)
        # Lower temperature sharpens the distribution → larger cross-entropy
        assert loss_low_temp > loss_high_temp


# ---------------------------------------------------------------------------
# prior_weighted_mse_loss
# ---------------------------------------------------------------------------

class TestPriorWeightedMSE:
    """Tests for the WAXS prior-weighted reconstruction loss L_MSE,W·G."""

    def test_peak_weighting_increases_loss(self):
        """Upweighting peak positions must produce a higher loss than plain MSE."""
        x_hat  = torch.zeros(BATCH_SIZE, WAXS_DIM)
        target = torch.ones(BATCH_SIZE, WAXS_DIM)

        # Plain MSE (weight 1 everywhere)
        loss_uniform = prior_weighted_mse_loss(x_hat, target, peak_positions=[], peak_weight=5.0)
        # Weighted MSE (peak positions upweighted)
        loss_weighted = prior_weighted_mse_loss(x_hat, target, peak_positions=[50, 150], peak_weight=5.0)
        assert loss_weighted > loss_uniform, "Peak weighting did not increase loss."


# ---------------------------------------------------------------------------
# TotalLoss
# ---------------------------------------------------------------------------

class TestTotalLoss:
    """Integration tests for the composite TotalLoss module."""

    @pytest.fixture()
    def loss_inputs(self) -> dict:
        """Return a dict of random tensors matching the TotalLoss forward signature."""
        return {
            "f_bg_s":   torch.abs(torch.randn(BATCH_SIZE, SAXS_DIM)),
            "f_bg_w":   torch.abs(torch.randn(BATCH_SIZE, WAXS_DIM)),
            "xball_s":  torch.abs(torch.randn(BATCH_SIZE, SAXS_DIM)),
            "xball_w":  torch.abs(torch.randn(BATCH_SIZE, WAXS_DIM)),
            "xhat_s":   torch.abs(torch.randn(BATCH_SIZE, SAXS_DIM)),
            "xhat_w":   torch.abs(torch.randn(BATCH_SIZE, WAXS_DIM)),
            "xdiff_s":  torch.randn(BATCH_SIZE, SAXS_DIM),
            "xdiff_w":  torch.randn(BATCH_SIZE, WAXS_DIM),
            "z_s":      torch.randn(BATCH_SIZE, LATENT_DIM),
            "z_w":      torch.randn(BATCH_SIZE, LATENT_DIM),
        }

    def test_total_loss_is_non_negative(self, loss_inputs: dict):
        """The weighted sum L_Total must be ≥ 0."""
        breakdown = TotalLoss()(**loss_inputs)
        assert breakdown.total.item() >= 0, f"L_Total was negative: {breakdown.total.item()}"

    def test_breakdown_fields_are_detached(self, loss_inputs: dict):
        """Individual loss components should not carry gradient history."""
        breakdown = TotalLoss()(**loss_inputs)
        for field_name in ("bg", "mse_s", "mse_w", "nce"):
            val = getattr(breakdown, field_name)
            assert not val.requires_grad, f"Field '{field_name}' still requires grad."

    def test_total_requires_grad_for_backward(self, loss_inputs: dict):
        """breakdown.total must be differentiable so backward() can be called."""
        breakdown = TotalLoss()(**loss_inputs)
        assert breakdown.total.requires_grad, "breakdown.total does not require grad."
        breakdown.total.backward()   # must not raise
