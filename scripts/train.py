"""Train the ScatteringModel.

Usage:
    python scripts/train.py --data data/raw/spectra.h5
    python scripts/train.py --data data/raw/spectra.h5 --override training.epochs=50
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config                   import load_config
from src.data.dataset             import build_dataloaders
from src.models.scattering_model  import ScatteringModel
from src.losses.losses            import TotalLoss
from src.training.trainer         import Trainer
from src.utils.logging_utils      import setup_logging

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/default.yaml")
    p.add_argument("--data",     required=True)
    p.add_argument("--resume",   default=None)
    p.add_argument("--log-file", default=None)
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args()
    setup_logging(log_file=args.log_file)
    cfg = load_config(args.config, overrides=args.override)
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        args.data, cfg.data.saxs_dim, cfg.data.waxs_dim,
        cfg.preprocessing.ball_radius, cfg.preprocessing.normalize,
        cfg.preprocessing.norm_mode,
        cfg.data.train_split, cfg.data.val_split,
        cfg.training.batch_size, seed=cfg.data.seed)
    model     = ScatteringModel.from_config(cfg)
    criterion = TotalLoss(cfg.loss.omega1, cfg.loss.omega2, cfg.loss.omega3, cfg.loss.omega4,
                          cfg.loss.lambda1, cfg.loss.lambda2,
                          cfg.loss.waxs_peak_positions, cfg.loss.waxs_peak_weight,
                          cfg.cross_modal.temperature)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    trainer = Trainer(model, criterion, cfg)
    trainer.fit(train_loader, val_loader, resume_path=args.resume)
    test = trainer._epoch(test_loader, False, -1)
    print(f"Test loss: {test['total']:.4f}")

if __name__ == "__main__": main()
