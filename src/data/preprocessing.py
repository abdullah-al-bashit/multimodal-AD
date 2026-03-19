"""Split, rolling-ball background, and normalisation for scattering spectra."""
from __future__ import annotations
import logging
from typing import Tuple
import numpy as np
from scipy.ndimage import grey_opening

logger = logging.getLogger(__name__)

def split_spectrum(x: np.ndarray, saxs_dim: int = 300, waxs_dim: int = 290) -> Tuple[np.ndarray, np.ndarray]:
    total = saxs_dim + waxs_dim
    if x.shape[-1] != total:
        raise ValueError(f"Expected last dim {total}, got {x.shape[-1]}.")
    return x[..., :saxs_dim], x[..., saxs_dim:]

def rolling_ball_background(x: np.ndarray, radius: int = 20) -> np.ndarray:
    struct = np.ones(2 * radius + 1)
    if x.ndim == 1:
        return grey_opening(x, structure=struct).astype(x.dtype)
    result = np.empty_like(x)
    for i in range(x.shape[0]):
        result[i] = grey_opening(x[i], structure=struct)
    return result

class SignalNormalizer:
    def __init__(self, mode: str = "zscore") -> None:
        if mode not in ("zscore", "minmax"):
            raise ValueError(f"mode must be 'zscore' or 'minmax', got '{mode}'")
        self.mode = mode
        self._loc = self._scale = None

    def fit(self, x: np.ndarray) -> "SignalNormalizer":
        if self.mode == "zscore":
            self._loc   = x.mean(axis=0)
            self._scale = x.std(axis=0) + 1e-8
        else:
            self._loc   = x.min(axis=0)
            self._scale = (x.max(axis=0) - x.min(axis=0)) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._loc is None: raise RuntimeError("Call fit() first.")
        return (x - self._loc) / self._scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self._loc is None: raise RuntimeError("Call fit() first.")
        return x * self._scale + self._loc

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

class ScatteringPreprocessor:
    """End-to-end preprocessing: split → rolling-ball BG → normalise."""
    def __init__(self, saxs_dim=300, waxs_dim=290, ball_radius=20,
                 normalize=True, norm_mode="zscore") -> None:
        self.saxs_dim    = saxs_dim
        self.waxs_dim    = waxs_dim
        self.ball_radius = ball_radius
        self.normalize   = normalize
        self._norm_s = SignalNormalizer(norm_mode) if normalize else None
        self._norm_w = SignalNormalizer(norm_mode) if normalize else None

    def fit_transform(self, x):
        return self._process(x, fit=True)

    def transform(self, x):
        return self._process(x, fit=False)

    def _process(self, x, fit):
        xs, xw     = split_spectrum(x, self.saxs_dim, self.waxs_dim)
        xball_s    = rolling_ball_background(xs, self.ball_radius)
        xball_w    = rolling_ball_background(xw, self.ball_radius)
        if self.normalize:
            xs = self._norm_s.fit_transform(xs) if fit else self._norm_s.transform(xs)
            xw = self._norm_w.fit_transform(xw) if fit else self._norm_w.transform(xw)
        return xs, xw, xball_s, xball_w
