"""
preprocess.py
=============
Offline preprocessing script: splits raw data and saves train/val/test HDF5 files.

Motivation
----------
Running the :class:`~src.data.preprocessing.ScatteringPreprocessor` inside the
DataLoader at training time is convenient but re-computes rolling-ball backgrounds
on every epoch.  This script pre-computes and caches the preprocessed arrays once,
so the training DataLoader can read them directly from compressed HDF5 files —
reducing CPU load and ensuring exact reproducibility across runs.

Output structure
----------------
For each split the script writes one HDF5 file with four gzip-compressed datasets:

::

    <output_dir>/train.h5
        /xs       (N_train, saxs_dim)  float32
        /xw       (N_train, waxs_dim)  float32
        /xball_s  (N_train, saxs_dim)  float32
        /xball_w  (N_train, waxs_dim)  float32
        /labels   (N_train,)           int64      # only if source has labels

Usage
-----
::

    python scripts/preprocess.py \\
        --input  data/raw/spectra.h5 \\
        --output data/processed/ \\
        --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

# Ensure the project root is on sys.path when running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import load_raw_data
from src.data.preprocessing import ScatteringPreprocessor
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with all argument values.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess raw scattering spectra and cache as HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the raw spectra file (.h5 or .npy).",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output", default="data/processed/",
        help="Directory where the processed HDF5 files will be saved.",
    )
    return parser.parse_args()


def _save_split(
    output_dir: Path,
    split_name: str,
    xs: np.ndarray,
    xw: np.ndarray,
    xball_s: np.ndarray,
    xball_w: np.ndarray,
    labels: np.ndarray | None,
) -> None:
    """Write one preprocessed split to a compressed HDF5 file.

    Args:
        output_dir: Directory to write the file into.
        split_name: One of ``"train"``, ``"val"``, or ``"test"``.
        xs:        Preprocessed SAXS signals,          shape ``(N, saxs_dim)``.
        xw:        Preprocessed WAXS signals,          shape ``(N, waxs_dim)``.
        xball_s:   SAXS rolling-ball BG,               shape ``(N, saxs_dim)``.
        xball_w:   WAXS rolling-ball BG,               shape ``(N, waxs_dim)``.
        labels:    Integer class labels ``(N,)`` or ``None``.
    """
    out_path = output_dir / f"{split_name}.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("xs",      data=xs,      compression="gzip")
        f.create_dataset("xw",      data=xw,      compression="gzip")
        f.create_dataset("xball_s", data=xball_s, compression="gzip")
        f.create_dataset("xball_w", data=xball_w, compression="gzip")
        if labels is not None:
            f.create_dataset("labels", data=labels, compression="gzip")

    logger.info(
        "Saved %s split → %s  (%d samples)", split_name, out_path, len(xs)
    )


def main() -> None:
    """Preprocessing entry point: load, split, preprocess, and save."""
    args = _parse_args()
    setup_logging()
    cfg = load_config(args.config)

    # ---- Load raw data -------------------------------------------------------
    X, labels = load_raw_data(args.input)
    num_samples = len(X)
    logger.info("Loaded %d spectra from '%s'", num_samples, args.input)

    # ---- Reproducible random split ------------------------------------------
    rng = np.random.default_rng(cfg.data.seed)
    indices = rng.permutation(num_samples)

    n_train = int(num_samples * cfg.data.train_split)
    n_val   = int(num_samples * cfg.data.val_split)

    splits = {
        "train": indices[:n_train],
        "val":   indices[n_train : n_train + n_val],
        "test":  indices[n_train + n_val :],
    }

    # ---- Preprocessing -------------------------------------------------------
    preprocessor = ScatteringPreprocessor(
        saxs_dim=cfg.data.saxs_dim,
        waxs_dim=cfg.data.waxs_dim,
        ball_radius=cfg.preprocessing.ball_radius,
        normalize=cfg.preprocessing.normalize,
        norm_mode=cfg.preprocessing.norm_mode,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, idx_arr in splits.items():
        # Fit only on training data to prevent leakage into val/test
        if split_name == "train":
            xs, xw, xball_s, xball_w = preprocessor.fit_transform(X[idx_arr])
        else:
            xs, xw, xball_s, xball_w = preprocessor.transform(X[idx_arr])

        split_labels = labels[idx_arr] if labels is not None else None

        _save_split(
            output_dir, split_name,
            xs, xw, xball_s, xball_w, split_labels,
        )

    logger.info("All splits saved to '%s/'", output_dir)


if __name__ == "__main__":
    main()
