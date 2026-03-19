"""PyTorch Dataset and DataLoader factory for SAXS/WAXS spectra."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocessing import ScatteringPreprocessor

logger = logging.getLogger(__name__)

class ScatteringDataset(Dataset):
    def __init__(self, xs, xw, xball_s, xball_w, labels=None) -> None:
        super().__init__()
        n = xs.shape[0]
        self.xs      = torch.from_numpy(xs).float()
        self.xw      = torch.from_numpy(xw).float()
        self.xball_s = torch.from_numpy(xball_s).float()
        self.xball_w = torch.from_numpy(xball_w).float()
        self.labels  = (torch.from_numpy(labels).long() if labels is not None
                        else torch.full((n,), -1, dtype=torch.long))

    def __len__(self): return len(self.xs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return dict(xs=self.xs[idx], xw=self.xw[idx],
                    xball_s=self.xball_s[idx], xball_w=self.xball_w[idx],
                    label=self.labels[idx],
                    index=torch.tensor(idx, dtype=torch.long))

def load_raw_data(path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    p = Path(path)
    if p.suffix in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            X      = f["spectra"][:]
            labels = f["labels"][:] if "labels" in f else None
        logger.info("Loaded HDF5 %s  shape=%s", p.name, X.shape)
        return X, labels
    if p.suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        logger.info("Loaded NPY %s  shape=%s", p.name, arr.shape)
        return arr, None
    raise ValueError(f"Unsupported format: {p.suffix}")

def build_dataloaders(data_path, saxs_dim=300, waxs_dim=290, ball_radius=20,
                      normalize=True, norm_mode="zscore",
                      train_split=0.70, val_split=0.15,
                      batch_size=32, num_workers=4, seed=42):
    X, labels = load_raw_data(data_path)
    n   = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(n * train_split)
    n_vl = int(n * val_split)
    i_tr, i_vl, i_ts = idx[:n_tr], idx[n_tr:n_tr+n_vl], idx[n_tr+n_vl:]

    prep = ScatteringPreprocessor(saxs_dim, waxs_dim, ball_radius, normalize, norm_mode)
    def _lbl(i): return labels[i] if labels is not None else None

    ds_tr = ScatteringDataset(*prep.fit_transform(X[i_tr]), _lbl(i_tr))
    ds_vl = ScatteringDataset(*prep.transform(X[i_vl]),     _lbl(i_vl))
    ds_ts = ScatteringDataset(*prep.transform(X[i_ts]),     _lbl(i_ts))

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    logger.info("Splits → train:%d  val:%d  test:%d", len(ds_tr), len(ds_vl), len(ds_ts))
    return (DataLoader(ds_tr, shuffle=True,  **kw),
            DataLoader(ds_vl, shuffle=False, **kw),
            DataLoader(ds_ts, shuffle=False, **kw),
            prep)
