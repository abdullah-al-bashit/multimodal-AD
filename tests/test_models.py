"""
test_models.py
==============
Unit and integration tests for all model sub-modules.

Covers:
  * :class:`~src.models.background_cnn.BackgroundCNN` output shape and
    non-negativity (Softplus output).
  * :class:`~src.models.decoder.Decoder` output shape and non-negativity.
  * :class:`~src.models.cross_modal.CrossModalFusion` for all three fusion
    strategies (attention, concat, add).
  * :class:`~src.models.scattering_model.ScatteringModel` full forward pass:
    tensor shapes, gradient flow, and parameter count sanity check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Allow running tests directly from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.background_cnn import BackgroundCNN
from src.models.cross_modal import CrossModalFusion
from src.models.decoder import Decoder
from src.models.scattering_model import ScatteringModel

# Shared batch size used across all tests
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# BackgroundCNN
# ---------------------------------------------------------------------------

class TestBackgroundCNN:
    """Tests for the per-modality background estimation CNN."""

    def test_output_shape(self):
        """BackgroundCNN must return a tensor of shape (B, signal_len)."""
        cnn = BackgroundCNN(signal_len=300)
        out = cnn(torch.randn(BATCH_SIZE, 300), torch.randn(BATCH_SIZE, 300))
        assert out.shape == (BATCH_SIZE, 300), f"Wrong shape: {out.shape}"

    def test_output_is_non_negative(self):
        """Softplus activation ensures the background estimate is always ≥ 0."""
        cnn = BackgroundCNN(signal_len=300)
        out = cnn(torch.randn(BATCH_SIZE, 300), torch.randn(BATCH_SIZE, 300))
        assert (out >= 0).all(), "BackgroundCNN produced negative values."


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TestDecoder:
    """Tests for the per-modality MLP decoder."""

    @pytest.mark.parametrize("out_dim", [300, 290])
    def test_output_shape_and_non_negativity(self, out_dim: int):
        """Decoder must return (B, out_dim) with non-negative values (Softplus)."""
        decoder = Decoder(latent_dim=64, out_dim=out_dim)
        out = decoder(torch.randn(BATCH_SIZE, 64))
        assert out.shape == (BATCH_SIZE, out_dim), f"Wrong shape: {out.shape}"
        assert (out >= 0).all(), "Decoder produced negative output values."


# ---------------------------------------------------------------------------
# CrossModalFusion
# ---------------------------------------------------------------------------

class TestCrossModalFusion:
    """Tests for the cross-modal fusion module."""

    @pytest.mark.parametrize("method", ["attention", "concat", "add"])
    def test_output_shape_all_fusion_methods(self, method: str):
        """All fusion strategies must output shape (B, latent_dim)."""
        fusion = CrossModalFusion(latent_dim=64, method=method)
        z_s = torch.randn(BATCH_SIZE, 64)
        z_w = torch.randn(BATCH_SIZE, 64)
        out = fusion(z_s, z_w)
        assert out.shape == (BATCH_SIZE, 64), (
            f"Method '{method}': wrong output shape {out.shape}"
        )


# ---------------------------------------------------------------------------
# ScatteringModel (end-to-end)
# ---------------------------------------------------------------------------

class TestScatteringModel:
    """Integration tests for the full end-to-end model."""

    @pytest.fixture()
    def small_model(self) -> ScatteringModel:
        """Return a lightweight ScatteringModel suitable for fast CPU tests."""
        return ScatteringModel(
            saxs_dim=300,
            waxs_dim=290,
            gnn_hidden=32,
            gnn_latent=16,
            gnn_layers=2,
            gnn_k=4,
        )

    def test_output_z_mp_shape(self, small_model: ScatteringModel):
        """Fused embedding Z_mp must have shape (B, latent_dim)."""
        out = small_model(
            torch.randn(BATCH_SIZE, 300), torch.randn(BATCH_SIZE, 290),
            torch.randn(BATCH_SIZE, 300), torch.randn(BATCH_SIZE, 290),
        )
        assert out.z_mp.shape == (BATCH_SIZE, 16)

    def test_output_xhat_shapes(self, small_model: ScatteringModel):
        """Reconstructed difference signals must match modality dimensions."""
        out = small_model(
            torch.randn(BATCH_SIZE, 300), torch.randn(BATCH_SIZE, 290),
            torch.randn(BATCH_SIZE, 300), torch.randn(BATCH_SIZE, 290),
        )
        assert out.xhat_s.shape == (BATCH_SIZE, 300)
        assert out.xhat_w.shape == (BATCH_SIZE, 290)

    def test_gradients_flow_through_model(self, small_model: ScatteringModel):
        """A backward pass must populate gradients for all trainable parameters."""
        out = small_model(
            torch.randn(BATCH_SIZE, 300), torch.randn(BATCH_SIZE, 290),
            torch.randn(BATCH_SIZE, 300), torch.randn(BATCH_SIZE, 290),
        )
        loss = out.z_mp.sum()
        loss.backward()

        # Every leaf parameter with requires_grad=True should have a gradient
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for '{name}'."
