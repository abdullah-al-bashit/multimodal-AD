"""Visualization helpers for scattering spectra and model outputs."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(signal, background=None, diff=None, title="Spectrum", save_path=None):
    fig, axes = plt.subplots(1, 2 if diff is not None else 1, figsize=(12,4))
    ax = axes[0] if diff is not None else axes
    ax.plot(signal, label="Signal", lw=1.2)
    if background is not None: ax.plot(background, "--", label="f_bg", lw=1.2)
    ax.set_title(title); ax.set_xlabel("Channel"); ax.set_ylabel("Intensity")
    ax.legend(); ax.grid(alpha=0.3)
    if diff is not None:
        axes[1].plot(diff, color="tab:orange", lw=1.2)
        axes[1].set_title(f"{title} — BG subtracted"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_reconstruction(original, reconstructed, title="Reconstruction", save_path=None):
    plt.figure(figsize=(8,4))
    plt.plot(original,      label="Target",        lw=1.2)
    plt.plot(reconstructed, label="Reconstructed", lw=1.2, linestyle="--")
    plt.title(title); plt.xlabel("Channel"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_latent_tsne(embeddings, labels=None, title="Latent Space (t-SNE)", save_path=None):
    from sklearn.manifold import TSNE
    z2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(7,6))
    sc = plt.scatter(z2d[:,0], z2d[:,1],
                     c=labels if labels is not None else "steelblue",
                     cmap="tab10", s=10, alpha=0.7)
    if labels is not None: plt.colorbar(sc, label="Class")
    plt.title(title); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_loss_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(7,4))
    plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Curves")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
