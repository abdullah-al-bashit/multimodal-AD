"""
preprocessing.py
================
Signal preprocessing utilities for combined SAXS/WAXS scattering spectra.

Pipeline
--------
1. **Split** – divide a concatenated 590-point spectrum into SAXS (300 pts)
   and WAXS (290 pts) sub-signals.
2. **Rolling-ball background** – estimate the smooth, slowly-varying
   background using a morphological grey-opening (equivalent to the
   rolling-ball algorithm).  The result ``x_ball`` is later used both as a
   guide signal for the background CNN and as the L_BG loss target.
3. **Normalise** – optionally standardise each modality (z-score or
   min-max) so that the two very different intensity scales are comparable
   before entering the neural network.

All operations are NumPy/SciPy based and run on CPU.  The outputs are
passed to ``ScatteringDataset`` which converts them to float32 tensors.

References
----------
- Rolling-ball / grey-opening background:
    Sternberg (1983), Biomedical Image Processing, IEEE Computer 16(1).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import grey_opening

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal splitting
# ---------------------------------------------------------------------------

def split_spectrum(
    x: np.ndarray,
    saxs_dim: int = 300,
    waxs_dim: int = 290,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a concatenated scattering spectrum into SAXS and WAXS portions.

    The raw instrument output is stored as a single flat array where the
    first ``saxs_dim`` points are the small-angle region and the remaining
    ``waxs_dim`` points are the wide-angle region.

    Args:
        x: Input array of shape ``(..., saxs_dim + waxs_dim)``.
        saxs_dim: Number of SAXS data points (default 300).
        waxs_dim: Number of WAXS data points (default 290).

    Returns:
        Tuple ``(xs, xw)`` where *xs* has shape ``(..., saxs_dim)`` and
        *xw* has shape ``(..., waxs_dim)``.

    Raises:
        ValueError: If the last dimension of *x* does not equal
            ``saxs_dim + waxs_dim``.
    """
    expected_total = saxs_dim + waxs_dim
    if x.shape[-1] != expected_total:
        raise ValueError(
            f"Expected last dimension {expected_total} "
            f"(saxs_dim={saxs_dim} + waxs_dim={waxs_dim}), "
            f"but got {x.shape[-1]}."
        )
    xs = x[..., :saxs_dim]
    xw = x[..., saxs_dim:]
    return xs, xw


# ---------------------------------------------------------------------------
# Rolling-ball background estimation
# ---------------------------------------------------------------------------

def rolling_ball_background(x: np.ndarray, radius: int = 20) -> np.ndarray:
    """Estimate the smooth background via morphological grey-opening.

    Grey-opening with a flat structuring element of width ``2*radius+1`` is
    mathematically equivalent to the 1-D rolling-ball algorithm: it removes
    sharp peaks while following the slowly varying baseline.

    The result ``x_ball`` serves two roles in the training pipeline:

    * **Input guide** to ``BackgroundCNN`` (stacked channel alongside the
      raw signal).
    * **Loss target** for the rolling-ball guide term in ``L_BG``:
      ``‖f_bg − x_ball‖²``.

    Args:
        x: Signal array of shape ``(L,)`` (single sample) or
           ``(N, L)`` (batch of N samples).
        radius: Half-width of the flat structuring element in data-point
            units.  Larger values yield a smoother, lower-lying background
            estimate.

    Returns:
        Background estimate with the same shape and dtype as *x*.
    """
    # Flat 1-D structuring element: ─────●───── length = 2*radius + 1
    structure = np.ones(2 * radius + 1)

    if x.ndim == 1:
        # Single spectrum: apply grey_opening directly
        return grey_opening(x, structure=structure).astype(x.dtype)

    # Batch of spectra: apply row-wise to keep memory footprint low
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        result[i] = grey_opening(x[i], structure=structure)
    return result


# ---------------------------------------------------------------------------
# Signal normalisation
# ---------------------------------------------------------------------------

class SignalNormalizer:
    """Fit-then-transform normaliser for 1-D scattering spectra.

    Supports two modes:

    * ``"zscore"`` – subtract the per-feature mean and divide by the
      per-feature standard deviation (with a small epsilon for stability).
    * ``"minmax"`` – scale each feature to [0, 1] using the training-set
      min/max.

    Parameters are estimated on the training split via :meth:`fit` and then
    reused for validation/test splits via :meth:`transform`, exactly like
    ``sklearn.preprocessing.StandardScaler``.

    Args:
        mode: Normalisation mode – ``"zscore"`` (default) or ``"minmax"``.

    Raises:
        ValueError: If *mode* is not one of the supported options.
    """

    _SUPPORTED_MODES = frozenset({"zscore", "minmax"})

    def __init__(self, mode: str = "zscore") -> None:
        if mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"mode must be one of {self._SUPPORTED_MODES}, got '{mode}'."
            )
        self.mode = mode
        # Set during fit(); None signals that fit() has not been called yet
        self._loc: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, x: np.ndarray) -> "SignalNormalizer":
        """Compute and store normalisation statistics from *x*.

        Args:
            x: 2-D array of shape ``(N, L)`` (training split only).

        Returns:
            ``self`` – allows method chaining: ``normalizer.fit(x).transform(x_val)``.
        """
        if self.mode == "zscore":
            self._loc = x.mean(axis=0)
            # Add epsilon to guard against zero-variance features
            self._scale = x.std(axis=0) + 1e-8
        else:  # minmax
            self._loc = x.min(axis=0)
            self._scale = (x.max(axis=0) - x.min(axis=0)) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the previously fitted normalisation to *x*.

        Args:
            x: Array of shape ``(..., L)`` – same feature dimension as
               the training data used in :meth:`fit`.

        Returns:
            Normalised array of the same shape as *x*.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        self._check_fitted()
        return (x - self._loc) / self._scale  # type: ignore[operator]

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Reverse the normalisation to recover original-scale values.

        Useful for converting model predictions back to physical intensity
        units after inference.

        Args:
            x: Normalised array of shape ``(..., L)``.

        Returns:
            Array in the original (unnormalised) scale.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        self._check_fitted()
        return x * self._scale + self._loc  # type: ignore[operator]

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Convenience wrapper: fit on *x* and immediately transform it.

        Should **only** be called on the training split.

        Args:
            x: Training data array of shape ``(N, L)``.

        Returns:
            Normalised version of *x*.
        """
        return self.fit(x).transform(x)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise if the normaliser has not been fitted yet."""
        if self._loc is None:
            raise RuntimeError(
                "SignalNormalizer has not been fitted. Call fit() first."
            )


# ---------------------------------------------------------------------------
# End-to-end preprocessing pipeline
# ---------------------------------------------------------------------------

class ScatteringPreprocessor:
    """End-to-end preprocessing pipeline: split → rolling-ball BG → normalise.

    This class is the single entry point for converting raw instrument data
    into the four tensors consumed by ``ScatteringDataset``:

    ============  ======================  ===================================
    Output        Shape                   Description
    ============  ======================  ===================================
    ``xs``        ``(N, saxs_dim)``       Normalised SAXS signal
    ``xw``        ``(N, waxs_dim)``       Normalised WAXS signal
    ``xball_s``   ``(N, saxs_dim)``       SAXS rolling-ball background guide
    ``xball_w``   ``(N, waxs_dim)``       WAXS rolling-ball background guide
    ============  ======================  ===================================

    Usage::

        prep = ScatteringPreprocessor()
        xs_tr, xw_tr, xball_s_tr, xball_w_tr = prep.fit_transform(X_train)
        xs_vl, xw_vl, xball_s_vl, xball_w_vl = prep.transform(X_val)

    Args:
        saxs_dim: Expected number of SAXS points (default 300).
        waxs_dim: Expected number of WAXS points (default 290).
        ball_radius: Radius for the rolling-ball background (default 20).
        normalize: Whether to normalise the signals (default ``True``).
        norm_mode: Normalisation mode – ``"zscore"`` or ``"minmax"``.
    """

    def __init__(
        self,
        saxs_dim: int = 300,
        waxs_dim: int = 290,
        ball_radius: int = 20,
        normalize: bool = True,
        norm_mode: str = "zscore",
    ) -> None:
        self.saxs_dim = saxs_dim
        self.waxs_dim = waxs_dim
        self.ball_radius = ball_radius
        self.normalize = normalize
        # Separate normaliser instances so SAXS and WAXS statistics are
        # computed independently (they have very different intensity ranges)
        self._norm_s = SignalNormalizer(norm_mode) if normalize else None
        self._norm_w = SignalNormalizer(norm_mode) if normalize else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit normalisation statistics on *x* then transform it.

        Must be called on the **training split only**.

        Args:
            x: Raw spectra of shape ``(N, saxs_dim + waxs_dim)``.

        Returns:
            Four arrays ``(xs, xw, xball_s, xball_w)`` – see class docstring.
        """
        return self._process(x, fit=True)

    def transform(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply previously fitted statistics to *x* (validation / test split).

        Args:
            x: Raw spectra of shape ``(N, saxs_dim + waxs_dim)``.

        Returns:
            Four arrays ``(xs, xw, xball_s, xball_w)`` – see class docstring.
        """
        return self._process(x, fit=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process(
        self, x: np.ndarray, fit: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Core processing logic shared by fit_transform and transform.

        Args:
            x: Raw spectra array of shape ``(N, saxs_dim + waxs_dim)``.
            fit: If ``True``, fit the normalisers before transforming.

        Returns:
            Tuple ``(xs, xw, xball_s, xball_w)``.
        """
        # Step 1: split into modalities
        xs, xw = split_spectrum(x, self.saxs_dim, self.waxs_dim)

        # Step 2: rolling-ball background computed *before* normalisation so
        # that the guide signal remains in the original intensity scale
        xball_s = rolling_ball_background(xs, self.ball_radius)
        xball_w = rolling_ball_background(xw, self.ball_radius)

        # Step 3: normalise the raw signals (background guides are NOT
        # normalised – they are used as absolute-scale targets in L_BG)
        if self.normalize:
            xs = self._norm_s.fit_transform(xs) if fit else self._norm_s.transform(xs)  # type: ignore[union-attr]
            xw = self._norm_w.fit_transform(xw) if fit else self._norm_w.transform(xw)  # type: ignore[union-attr]

        return xs, xw, xball_s, xball_w
