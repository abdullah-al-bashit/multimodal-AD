"""Offline preprocessing → saves train/val/test HDF5 files.

Usage:
    python scripts/preprocess.py --input data/raw/spectra.h5
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import h5py, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config                  import load_config
from src.data.dataset            import load_raw_data
from src.data.preprocessing      import ScatteringPreprocessor
from src.utils.logging_utils     import setup_logging
logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output", default="data/processed/")
    args = p.parse_args(); setup_logging(); cfg = load_config(args.config)
    X, labels = load_raw_data(args.input)
    n   = len(X); rng = np.random.default_rng(cfg.data.seed); idx = rng.permutation(n)
    n_tr = int(n*cfg.data.train_split); n_vl = int(n*cfg.data.val_split)
    splits = dict(train=idx[:n_tr], val=idx[n_tr:n_tr+n_vl], test=idx[n_tr+n_vl:])
    prep = ScatteringPreprocessor(cfg.data.saxs_dim, cfg.data.waxs_dim,
                                  cfg.preprocessing.ball_radius, cfg.preprocessing.normalize,
                                  cfg.preprocessing.norm_mode)
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx_arr in splits.items():
        xs,xw,xbs,xbw = (prep.fit_transform if name=="train" else prep.transform)(X[idx_arr])
        with h5py.File(out_dir/f"{name}.h5","w") as f:
            f.create_dataset("xs",      data=xs,  compression="gzip")
            f.create_dataset("xw",      data=xw,  compression="gzip")
            f.create_dataset("xball_s", data=xbs, compression="gzip")
            f.create_dataset("xball_w", data=xbw, compression="gzip")
            if labels is not None: f.create_dataset("labels", data=labels[idx_arr])
        logger.info("Saved %s → %s (%d samples)", name, out_dir/f"{name}.h5", len(idx_arr))

if __name__ == "__main__": main()
