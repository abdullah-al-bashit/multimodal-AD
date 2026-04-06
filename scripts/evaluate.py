"""
evaluate.py
===========
Entry-point script for evaluating a trained model on the test split.

The script:
  1. Loads a trained checkpoint and reconstructs the :class:`~src.models.scattering_model.ScatteringModel`.
  2. Runs inference on the test DataLoader.
  3. Optionally plots the fused t-SNE embedding and a sample SAXS reconstruction.
  4. Saves all embedding arrays (``z_mp``, ``z_s``, ``z_w``, ``labels``) to
     ``outputs/`` as NumPy ``.npy`` files.

Usage
-----
::

    python scripts/evaluate.py \\
        --data data/raw/spectra.h5 \\
        --ckpt checkpoints/checkpoint_best.pt \\
        --plot
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the project root is on sys.path when running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.dataset import build_dataloaders
from src.models.scattering_model import ScatteringModel, ModelOutput
from src.utils.logging_utils import setup_logging
from src.utils.visualization import plot_latent_tsne, plot_reconstruction

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with all argument values.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ScatteringModel on the test split.",
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
        "--ckpt", required=True,
        help="Path to the trained .pt checkpoint.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Display t-SNE embedding and a sample reconstruction plot.",
    )
    return parser.parse_args()


@torch.no_grad()
def extract_embeddings(
    model: ScatteringModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over the full DataLoader and concatenate output arrays.

    Args:
        model:  Trained :class:`~src.models.scattering_model.ScatteringModel`
                in evaluation mode.
        loader: DataLoader for the split to evaluate.
        device: Device on which to run inference.

    Returns:
        Six numpy arrays ``(z_mp, z_s, z_w, labels, xhat_s, xdiff_s)``
        each of shape ``(N, ...)``.
    """
    model.eval()

    z_mp_list:   list[np.ndarray] = []
    z_s_list:    list[np.ndarray] = []
    z_w_list:    list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    xhat_s_list: list[np.ndarray] = []
    xdiff_s_list: list[np.ndarray] = []

    for batch in loader:
        xs      = batch["xs"].to(device)
        xw      = batch["xw"].to(device)
        xball_s = batch["xball_s"].to(device)
        xball_w = batch["xball_w"].to(device)

        out: ModelOutput = model(xs, xw, xball_s, xball_w)

        z_mp_list.append(out.z_mp.cpu().numpy())
        z_s_list.append(out.z_s.cpu().numpy())
        z_w_list.append(out.z_w.cpu().numpy())
        labels_list.append(batch["label"].numpy())
        xhat_s_list.append(out.xhat_s.cpu().numpy())
        xdiff_s_list.append(out.xdiff_s.cpu().numpy())

    return (
        np.concatenate(z_mp_list),
        np.concatenate(z_s_list),
        np.concatenate(z_w_list),
        np.concatenate(labels_list),
        np.concatenate(xhat_s_list),
        np.concatenate(xdiff_s_list),
    )


def main() -> None:
    """Evaluation entry point: load checkpoint, infer, plot, and save outputs."""
    args = _parse_args()
    setup_logging()
    cfg = load_config(args.config)

    # Resolve inference device (fall back to CPU if CUDA unavailable)
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )
    logger.info("Inference device: %s", device)

    # ---- Data ----------------------------------------------------------------
    _, _, test_loader, _ = build_dataloaders(
        data_path=args.data,
        saxs_dim=cfg.data.saxs_dim,
        waxs_dim=cfg.data.waxs_dim,
        ball_radius=cfg.preprocessing.ball_radius,
        normalize=cfg.preprocessing.normalize,
        norm_mode=cfg.preprocessing.norm_mode,
        batch_size=cfg.training.batch_size,
        seed=cfg.data.seed,
    )

    # ---- Model + checkpoint --------------------------------------------------
    model = ScatteringModel.from_config(cfg)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    logger.info("Loaded checkpoint from '%s'", args.ckpt)

    # ---- Inference -----------------------------------------------------------
    z_mp, z_s, z_w, labels, xhat_s, xdiff_s = extract_embeddings(
        model, test_loader, device
    )
    logger.info("Extracted embeddings: z_mp=%s  z_s=%s  z_w=%s", z_mp.shape, z_s.shape, z_w.shape)

    # ---- Optional plots ------------------------------------------------------
    if args.plot:
        # Use labels only if all samples are labelled (no sentinel -1 values)
        display_labels = labels if (labels >= 0).all() else None
        plot_latent_tsne(z_mp, display_labels, title="Z_mp (fused cross-modal)")
        plot_reconstruction(
            xdiff_s[0], xhat_s[0], title="SAXS reconstruction – sample 0"
        )

    # ---- Save embeddings to disk ---------------------------------------------
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for array_name, array_data in [
        ("z_mp",   z_mp),
        ("z_s",    z_s),
        ("z_w",    z_w),
        ("labels", labels),
    ]:
        out_path = output_dir / f"{array_name}.npy"
        np.save(out_path, array_data)

    logger.info("Embeddings saved to '%s/'", output_dir)


if __name__ == "__main__":
    main()
