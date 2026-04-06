"""
dataset.py
==========
PyTorch Dataset and DataLoader factory for SAXS/WAXS scattering spectra.

Overview
--------
This module provides two public entry points:

1. :class:`ScatteringDataset` – a :class:`~torch.utils.data.Dataset` wrapping
   the four preprocessed arrays (``xs``, ``xw``, ``xball_s``, ``xball_w``)
   returned by :class:`~src.data.preprocessing.ScatteringPreprocessor`.

2. :func:`build_dataloaders` – convenience factory that reads a raw data file,
   splits it into train/val/test, runs preprocessing, and returns three
   :class:`~torch.utils.data.DataLoader` instances ready for the training loop.

Supported file formats
----------------------
* **HDF5** (``.h5`` / ``.hdf5``): expects datasets ``spectra`` (required) and
  optionally ``labels`` (integer class labels).
* **NumPy** (``.npy``): a plain 2-D array of spectra; labels are set to -1.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import ScatteringPreprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ScatteringDataset(Dataset):
    """In-memory dataset for SAXS/WAXS scattering spectra.

    Holds four pre-processed float32 tensor arrays and an optional integer
    label array.  All arrays must have the same number of samples *N*.

    Args:
        xs:      Normalised SAXS signals,          shape ``(N, saxs_dim)``.
        xw:      Normalised WAXS signals,          shape ``(N, waxs_dim)``.
        xball_s: SAXS rolling-ball BG guide,       shape ``(N, saxs_dim)``.
        xball_w: WAXS rolling-ball BG guide,       shape ``(N, waxs_dim)``.
        labels:  Integer class labels of shape ``(N,)``.  Pass ``None`` to
                 auto-fill with ``-1`` (unlabelled).
    """

    def __init__(
        self,
        xs: np.ndarray,
        xw: np.ndarray,
        xball_s: np.ndarray,
        xball_w: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        num_samples = xs.shape[0]

        # Convert numpy arrays to float32 tensors once at construction time
        self.xs      = torch.from_numpy(xs).float()
        self.xw      = torch.from_numpy(xw).float()
        self.xball_s = torch.from_numpy(xball_s).float()
        self.xball_w = torch.from_numpy(xball_w).float()

        # Use -1 as a sentinel for unlabelled samples
        self.labels = (
            torch.from_numpy(labels).long()
            if labels is not None
            else torch.full((num_samples,), fill_value=-1, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample as a dictionary of tensors.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
              - ``"xs"``      – SAXS signal,            shape ``(saxs_dim,)``.
              - ``"xw"``      – WAXS signal,            shape ``(waxs_dim,)``.
              - ``"xball_s"`` – SAXS rolling-ball BG,   shape ``(saxs_dim,)``.
              - ``"xball_w"`` – WAXS rolling-ball BG,   shape ``(waxs_dim,)``.
              - ``"label"``   – class label,             scalar long tensor.
              - ``"index"``   – original dataset index,  scalar long tensor.
        """
        return {
            "xs":      self.xs[idx],
            "xw":      self.xw[idx],
            "xball_s": self.xball_s[idx],
            "xball_w": self.xball_w[idx],
            "label":   self.labels[idx],
            "index":   torch.tensor(idx, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_raw_data(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a raw spectra array and optional labels from disk.

    Supported formats:

    * **HDF5** (``.h5`` / ``.hdf5``): ``spectra`` dataset required;
      ``labels`` dataset is optional.
    * **NumPy** (``.npy``): 1-D arrays are reshaped to ``(1, L)``.

    Args:
        path: Path to the data file.

    Returns:
        Tuple ``(X, labels)`` where *X* has shape ``(N, L)`` and *labels*
        is either ``(N,)`` or ``None``.

    Raises:
        ValueError: If the file extension is not supported.
    """
    p = Path(path)

    if p.suffix in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            X = f["spectra"][:]
            labels = f["labels"][:] if "labels" in f else None
        logger.info("Loaded HDF5 '%s'  shape=%s", p.name, X.shape)
        return X, labels

    if p.suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
        # Handle single-sample files stored as a 1-D vector
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        logger.info("Loaded NPY '%s'  shape=%s", p.name, arr.shape)
        return arr, None

    raise ValueError(
        f"Unsupported file format '{p.suffix}'. "
        "Expected '.h5', '.hdf5', or '.npy'."
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_path: str | Path,
    saxs_dim: int = 300,
    waxs_dim: int = 290,
    ball_radius: int = 20,
    normalize: bool = True,
    norm_mode: str = "zscore",
    train_split: float = 0.70,
    val_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, ScatteringPreprocessor]:
    """Load data, preprocess, split, and return three DataLoaders.

    Steps:
      1. Read raw spectra from ``data_path`` via :func:`load_raw_data`.
      2. Randomly permute and split into train / val / test subsets.
      3. Fit :class:`~src.data.preprocessing.ScatteringPreprocessor` on the
         training split only, then transform all three splits (no data
         leakage).
      4. Wrap each split in a :class:`ScatteringDataset` and a
         :class:`~torch.utils.data.DataLoader`.

    Args:
        data_path: Path to the raw data file (.h5 or .npy).
        saxs_dim: SAXS signal length.
        waxs_dim: WAXS signal length.
        ball_radius: Rolling-ball radius for background estimation.
        normalize: Whether to normalise signals.
        norm_mode: Normalisation mode – ``"zscore"`` or ``"minmax"``.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        batch_size: Samples per mini-batch.
        num_workers: Number of DataLoader worker processes.
        seed: RNG seed for reproducible splitting.

    Returns:
        Tuple ``(train_loader, val_loader, test_loader, preprocessor)``
        where *preprocessor* can be used for ``inverse_transform`` at
        inference time.
    """
    X, labels = load_raw_data(data_path)
    num_samples = len(X)

    # Reproducible random permutation for dataset splitting
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    # Compute split boundary indices
    n_train = int(num_samples * train_split)
    n_val   = int(num_samples * val_split)
    idx_train = indices[:n_train]
    idx_val   = indices[n_train : n_train + n_val]
    idx_test  = indices[n_train + n_val :]

    # Helper to slice labels safely (returns None if labels is None)
    def _slice_labels(idx: np.ndarray) -> Optional[np.ndarray]:
        return labels[idx] if labels is not None else None

    # Preprocessing: fit ONLY on training split to prevent data leakage
    preprocessor = ScatteringPreprocessor(
        saxs_dim, waxs_dim, ball_radius, normalize, norm_mode
    )
    ds_train = ScatteringDataset(*preprocessor.fit_transform(X[idx_train]), _slice_labels(idx_train))
    ds_val   = ScatteringDataset(*preprocessor.transform(X[idx_val]),       _slice_labels(idx_val))
    ds_test  = ScatteringDataset(*preprocessor.transform(X[idx_test]),      _slice_labels(idx_test))

    logger.info(
        "Dataset splits – train: %d  val: %d  test: %d",
        len(ds_train), len(ds_val), len(ds_test),
    )

    # Shared DataLoader keyword arguments
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(ds_train, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(ds_val,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(ds_test,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, preprocessor
