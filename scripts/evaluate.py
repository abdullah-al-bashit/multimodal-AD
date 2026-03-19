"""Evaluate a trained model and export embeddings.

Usage:
    python scripts/evaluate.py --data data/raw/spectra.h5 \
        --ckpt checkpoints/checkpoint_best.pt --plot
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config                   import load_config
from src.data.dataset             import build_dataloaders
from src.models.scattering_model  import ScatteringModel
from src.utils.logging_utils      import setup_logging
from src.utils.visualization      import plot_latent_tsne, plot_reconstruction

@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    zmp, zs, zw, labs, xhs, xds = [],[],[],[],[],[]
    for b in loader:
        xs, xw = b["xs"].to(device), b["xw"].to(device)
        out = model(xs, xw, b["xball_s"].to(device), b["xball_w"].to(device))
        zmp.append(out.z_mp.cpu().numpy()); zs.append(out.z_s.cpu().numpy())
        zw.append(out.z_w.cpu().numpy());   labs.append(b["label"].numpy())
        xhs.append(out.xhat_s.cpu().numpy()); xds.append(out.xdiff_s.cpu().numpy())
    return (np.concatenate(zmp), np.concatenate(zs), np.concatenate(zw),
            np.concatenate(labs), np.concatenate(xhs), np.concatenate(xds))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data",   required=True)
    p.add_argument("--ckpt",   required=True)
    p.add_argument("--plot",   action="store_true")
    args = p.parse_args()
    setup_logging(); cfg = load_config(args.config)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = build_dataloaders(
        args.data, cfg.data.saxs_dim, cfg.data.waxs_dim,
        cfg.preprocessing.ball_radius, cfg.preprocessing.normalize,
        cfg.preprocessing.norm_mode, batch_size=cfg.training.batch_size, seed=cfg.data.seed)
    model = ScatteringModel.from_config(cfg)
    ck    = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model_state"]); model.to(device)
    z_mp, z_s, z_w, labels, xhat_s, xdiff_s = extract(model, test_loader, device)
    print(f"Embeddings: {z_mp.shape}")
    if args.plot:
        lbl = labels if (labels >= 0).all() else None
        plot_latent_tsne(z_mp, lbl, "Z_mp (fused)")
        plot_reconstruction(xdiff_s[0], xhat_s[0], "SAXS recon sample 0")
    out = Path("outputs"); out.mkdir(exist_ok=True)
    for name, arr in [("z_mp",z_mp),("z_s",z_s),("z_w",z_w),("labels",labels)]:
        np.save(out/f"{name}.npy", arr)
    print(f"Saved to {out}/")

if __name__ == "__main__": main()
