"""
train.py
========
Entry-point script for training the multimodal SAXS/WAXS scattering model.

The script:
  1. Parses CLI arguments.
  2. Loads configuration from a YAML file with optional dot-list overrides.
  3. Builds train / val / test DataLoaders.
  4. Constructs the :class:`~src.models.scattering_model.ScatteringModel`
     and :class:`~src.losses.losses.TotalLoss` from config.
  5. Runs the :class:`~src.training.trainer.Trainer` training loop.
  6. Evaluates the final model on the held-out test split.

Usage
-----
::

    # Basic run with default config
    python scripts/train.py --data data/raw/spectra.h5

    # Custom config + CLI overrides
    python scripts/train.py \\
        --data     data/raw/spectra.h5 \\
        --config   configs/experiment_01.yaml \\
        --override training.epochs=50 gnn.latent_dim=128 \\
        --log-file logs/train.log

    # Resume from a checkpoint
    python scripts/train.py \\
        --data   data/raw/spectra.h5 \\
        --resume checkpoints/checkpoint_epoch_0050.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import build_dataloaders
from src.losses.losses import TotalLoss
from src.models.scattering_model import ScatteringModel
from src.training.trainer import Trainer
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with all argument values.
    """
    parser = argparse.ArgumentParser(
        description="Train the multimodal SAXS/WAXS scattering model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the raw spectra file (.h5 or .npy).",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a .pt checkpoint to resume training from.",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Optional path to a log file (in addition to stdout).",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dot-list overrides, e.g. training.epochs=50.",
    )
    return parser.parse_args()


def main() -> None:
    """Training entry point: load config, build components, and run training."""
    args = _parse_args()
    setup_logging(log_file=args.log_file)

    # ---- Configuration -------------------------------------------------------
    cfg = load_config(args.config, overrides=args.override)

    # ---- Data ----------------------------------------------------------------
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        data_path=args.data,
        saxs_dim=cfg.data.saxs_dim,
        waxs_dim=cfg.data.waxs_dim,
        ball_radius=cfg.preprocessing.ball_radius,
        normalize=cfg.preprocessing.normalize,
        norm_mode=cfg.preprocessing.norm_mode,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        batch_size=cfg.training.batch_size,
        seed=cfg.data.seed,
    )

    # ---- Model and loss -------------------------------------------------------
    model = ScatteringModel.from_config(cfg)
    criterion = TotalLoss(
        omega1=cfg.loss.omega1,
        omega2=cfg.loss.omega2,
        omega3=cfg.loss.omega3,
        omega4=cfg.loss.omega4,
        lambda1=cfg.loss.lambda1,
        lambda2=cfg.loss.lambda2,
        waxs_peak_positions=cfg.loss.waxs_peak_positions,
        waxs_peak_weight=cfg.loss.waxs_peak_weight,
        temperature=cfg.cross_modal.temperature,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_params:,}")

    # ---- Training ------------------------------------------------------------
    trainer = Trainer(model, criterion, cfg)
    trainer.fit(train_loader, val_loader, resume_path=args.resume)

    # ---- Test evaluation -----------------------------------------------------
    test_metrics = trainer._run_epoch(test_loader, is_train=False, epoch=-1)
    logger.info(
        "Test results – total=%.4f  bg=%.4f  mse_s=%.4f  mse_w=%.4f  nce=%.4f",
        test_metrics["total"],
        test_metrics["bg"],
        test_metrics["mse_s"],
        test_metrics["mse_w"],
        test_metrics["nce"],
    )


if __name__ == "__main__":
    main()
