"""
visualization.py
================
Matplotlib helper functions for inspecting SAXS/WAXS spectra and model outputs.

All functions follow the same convention:
* Display the figure with ``plt.show()``.
* Optionally save a high-DPI PNG to *save_path* before showing.
* Return ``None`` – they are side-effect functions only.

Dependencies
------------
* ``matplotlib`` (always required)
* ``sklearn.manifold.TSNE`` (imported lazily in :func:`plot_latent_tsne` only)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum(
    signal: np.ndarray,
    background: Optional[np.ndarray] = None,
    diff: Optional[np.ndarray] = None,
    title: str = "Spectrum",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a raw spectrum with an optional background overlay and difference signal.

    If *diff* is provided, the figure has two subplots side-by-side:
    (1) the raw signal + background, and (2) the background-subtracted difference.

    Args:
        signal:     1-D array of the raw scattering intensity, shape ``(L,)``.
        background: Optional 1-D background estimate ``f_bg``, shape ``(L,)``.
        diff:       Optional 1-D difference signal ``signal − f_bg``, shape ``(L,)``.
        title:      Figure/subplot title string.
        save_path:  If provided, save the figure to this path at 150 DPI.
    """
    n_cols = 2 if diff is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(12, 4))

    # When diff is None, axes is a single Axes object, not an array
    ax_signal = axes[0] if diff is not None else axes

    ax_signal.plot(signal, label="Signal", lw=1.2)
    if background is not None:
        ax_signal.plot(background, "--", label="f_bg", lw=1.2)
    ax_signal.set_title(title)
    ax_signal.set_xlabel("Channel")
    ax_signal.set_ylabel("Intensity")
    ax_signal.legend()
    ax_signal.grid(alpha=0.3)

    if diff is not None:
        axes[1].plot(diff, color="tab:orange", lw=1.2)
        axes[1].set_title(f"{title} — BG subtracted")
        axes[1].set_xlabel("Channel")
        axes[1].set_ylabel("Intensity")
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    title: str = "Reconstruction",
    save_path: Optional[str | Path] = None,
) -> None:
    """Overlay the target and reconstructed difference signals on a single axis.

    Useful for visually assessing Decoder quality on individual samples.

    Args:
        original:      Ground-truth difference signal, shape ``(L,)``.
        reconstructed: Decoder output,                 shape ``(L,)``.
        title:         Plot title string.
        save_path:     If provided, save the figure to this path at 150 DPI.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(original,      label="Target",        lw=1.2)
    plt.plot(reconstructed, label="Reconstructed", lw=1.2, linestyle="--")
    plt.title(title)
    plt.xlabel("Channel")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_latent_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Latent Space (t-SNE)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Reduce embeddings to 2-D with t-SNE and produce a scatter plot.

    ``sklearn`` is imported lazily so that the rest of the module is usable
    without it when t-SNE visualisation is not needed.

    Args:
        embeddings: High-dimensional embedding matrix, shape ``(N, D)``.
        labels:     Optional integer class labels, shape ``(N,)``.  If provided,
                    points are coloured by class using the ``"tab10"`` colormap
                    and a colorbar is added.
        title:      Plot title string.
        save_path:  If provided, save the figure to this path at 150 DPI.
    """
    from sklearn.manifold import TSNE  # noqa: PLC0415 (lazy import)

    # Fit t-SNE; random_state ensures reproducibility across calls
    z2d: np.ndarray = TSNE(
        n_components=2, perplexity=30, random_state=42
    ).fit_transform(embeddings)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        z2d[:, 0],
        z2d[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab10",
        s=10,
        alpha=0.7,
    )
    if labels is not None:
        plt.colorbar(scatter, label="Class")
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot train and validation loss curves over epochs.

    Args:
        train_losses: List of per-epoch training losses.
        val_losses:   List of per-epoch validation losses.
        save_path:    If provided, save the figure to this path at 150 DPI.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
