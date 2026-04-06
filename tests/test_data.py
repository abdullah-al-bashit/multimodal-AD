"""
test_data.py
============
Unit tests for :mod:`src.data.preprocessing`.

Covers:
  * :func:`~src.data.preprocessing.split_spectrum` output shapes and error handling.
  * :func:`~src.data.preprocessing.rolling_ball_background` output shape and
    non-negativity guarantee.
  * :class:`~src.data.preprocessing.ScatteringPreprocessor` end-to-end
    ``fit_transform`` behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running tests directly from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import (
    ScatteringPreprocessor,
    rolling_ball_background,
    split_spectrum,
)


# ---------------------------------------------------------------------------
# split_spectrum
# ---------------------------------------------------------------------------

class TestSplitSpectrum:
    """Tests for the spectrum splitter utility."""

    def test_output_shapes_are_correct(self):
        """SAXS slice must be (N, 300) and WAXS (N, 290) for 590-wide input."""
        X = np.random.rand(10, 590)
        xs, xw = split_spectrum(X)
        assert xs.shape == (10, 300), f"Expected (10, 300), got {xs.shape}"
        assert xw.shape == (10, 290), f"Expected (10, 290), got {xw.shape}"

    def test_wrong_total_dimension_raises(self):
        """A spectrum of incorrect length must raise ValueError."""
        with pytest.raises(ValueError, match="total_dim"):
            split_spectrum(np.random.rand(10, 500))


# ---------------------------------------------------------------------------
# rolling_ball_background
# ---------------------------------------------------------------------------

class TestRollingBallBackground:
    """Tests for the morphological rolling-ball background estimator."""

    def test_output_shape_matches_input(self):
        """Background output must have the same shape as the input signal."""
        x = np.random.rand(8, 300)
        bg = rolling_ball_background(x, ball_radius=20)
        assert bg.shape == x.shape, f"Shape mismatch: {bg.shape} != {x.shape}"

    def test_output_is_non_negative(self):
        """Background must be non-negative for non-negative input signals."""
        x = np.abs(np.random.rand(4, 300))
        bg = rolling_ball_background(x, ball_radius=10)
        assert (bg >= 0).all(), "Background contains negative values."


# ---------------------------------------------------------------------------
# ScatteringPreprocessor
# ---------------------------------------------------------------------------

class TestScatteringPreprocessor:
    """Integration-level tests for the full preprocessing pipeline."""

    def test_fit_transform_output_shapes(self):
        """fit_transform must return four arrays with correct SAXS/WAXS shapes."""
        X = np.random.rand(50, 590).astype(np.float32)
        xs, xw, xball_s, xball_w = ScatteringPreprocessor().fit_transform(X)

        assert xs.shape      == (50, 300), f"xs shape wrong: {xs.shape}"
        assert xw.shape      == (50, 290), f"xw shape wrong: {xw.shape}"
        assert xball_s.shape == (50, 300), f"xball_s shape wrong: {xball_s.shape}"
        assert xball_w.shape == (50, 290), f"xball_w shape wrong: {xball_w.shape}"

    def test_transform_without_fit_raises(self):
        """Calling transform before fit must raise RuntimeError."""
        prep = ScatteringPreprocessor()
        with pytest.raises(RuntimeError, match="fit"):
            prep.transform(np.random.rand(5, 590).astype(np.float32))
