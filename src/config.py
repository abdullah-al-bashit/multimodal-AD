"""
config.py
=========
Centralised, type-safe configuration management using OmegaConf structured configs.

Design
------
Every hyper-parameter in the pipeline is captured in a hierarchy of frozen
Python dataclasses.  At runtime, OmegaConf merges a YAML file on top of the
dataclass defaults, then applies any command-line dot-list overrides.  The
final merged config is cast back to the typed dataclass tree so that
downstream code benefits from IDE auto-complete and type checking.

Usage example::

    from src.config import load_config

    cfg = load_config("configs/default.yaml", overrides=["training.epochs=50"])
    model = ScatteringModel.from_config(cfg)

YAML structure (mirrors the dataclass hierarchy)::

    data:
      saxs_dim: 300
      waxs_dim: 290
    training:
      epochs: 100
      batch_size: 32
      ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Paths and dataset-level hyper-parameters.

    Attributes:
        raw_dir: Directory containing the raw instrument HDF5/NPY files.
        processed_dir: Directory where preprocessed outputs are cached.
        saxs_dim: Number of SAXS data points (first segment of each spectrum).
        waxs_dim: Number of WAXS data points (second segment of each spectrum).
        total_dim: Total spectrum length; must equal ``saxs_dim + waxs_dim``.
        train_split: Fraction of data used for training (0–1).
        val_split: Fraction of data used for validation (0–1).
        test_split: Fraction of data used for test (0–1).
        seed: Random seed for reproducible data splitting.
    """

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    saxs_dim: int = 300
    waxs_dim: int = 290
    total_dim: int = 590          # validated: must equal saxs_dim + waxs_dim
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42


@dataclass
class PreprocessingConfig:
    """Signal preprocessing hyper-parameters.

    Attributes:
        ball_radius: Half-width of the rolling-ball structuring element in
            data-point units.  Controls the scale of background features
            that are removed.
        normalize: Whether to normalise signals to zero mean / unit variance
            (or min-max range) before feeding to the model.
        norm_mode: Normalisation strategy – ``"zscore"`` (default) or
            ``"minmax"``.
    """

    ball_radius: int = 20
    normalize: bool = True
    norm_mode: str = "zscore"


@dataclass
class BackgroundCNNConfig:
    """BackgroundCNN architecture hyper-parameters.

    Attributes:
        hidden_channels: Channel widths for the CNN stem and body.  The list
            length determines how many convolutional stages are built.
        kernel_size: Shared 1-D convolution kernel size (must be odd for
            symmetric same-padding).
        dropout: Dropout probability inside residual blocks.
    """

    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 64])
    kernel_size: int = 5
    dropout: float = 0.1


@dataclass
class GNNConfig:
    """GNNEncoder architecture hyper-parameters.

    Attributes:
        k_neighbors: Number of nearest neighbours for the k-NN graph.
        hidden_dim: Width of intermediate GATv2Conv layers.
        latent_dim: Dimensionality *D* of the output embedding.
        num_layers: Number of GATv2Conv message-passing layers.
        heads: Number of attention heads (all layers except the last).
        dropout: Dropout probability after each GATv2Conv layer.
    """

    k_neighbors: int = 8
    hidden_dim: int = 128
    latent_dim: int = 64
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.1


@dataclass
class CrossModalConfig:
    """Cross-modal fusion hyper-parameters.

    Attributes:
        method: Fusion strategy – ``"attention"`` (cross-MHA), ``"concat"``
            (MLP on concatenated embeddings), or ``"add"`` (element-wise sum).
        temperature: InfoNCE contrastive loss temperature ``τ``.  A smaller
            value sharpens the similarity distribution.
    """

    method: str = "attention"
    temperature: float = 0.07


@dataclass
class DecoderConfig:
    """Decoder MLP hyper-parameters.

    Attributes:
        hidden_dim: Width *H* of the first hidden layer (second layer uses 2H).
        dropout: Dropout probability in each MLP block.
    """

    hidden_dim: int = 128
    dropout: float = 0.1


@dataclass
class LossConfig:
    """Training loss weights and domain-specific parameters.

    Attributes:
        omega1: Weight for the background loss ``L_BG``.
        omega2: Weight for the SAXS reconstruction loss ``L_MSE,S``.
        omega3: Weight for the WAXS prior-weighted reconstruction loss ``L_MSE,W·G``.
        omega4: Weight for the InfoNCE contrastive loss ``L_NCE``.
        lambda1: Smoothness penalty weight inside ``L_BG``.
        lambda2: Positivity penalty weight inside ``L_BG``.
        waxs_peak_positions: Q-array indices of biologically significant WAXS
            features (e.g. amyloid β-sheet reflection) to upweight.
        waxs_peak_weight: Multiplier applied to squared errors at peak positions.
    """

    omega1: float = 1.0
    omega2: float = 1.0
    omega3: float = 1.0
    omega4: float = 0.5
    lambda1: float = 0.01
    lambda2: float = 0.1
    waxs_peak_positions: List[int] = field(default_factory=lambda: [50, 150])
    waxs_peak_weight: float = 5.0


@dataclass
class TrainingConfig:
    """Training loop hyper-parameters and I/O paths.

    Attributes:
        epochs: Total number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate for AdamW.
        weight_decay: L2 regularisation coefficient for AdamW.
        lr_scheduler: LR schedule type – ``"cosine"`` or ``"step"``.
        warmup_epochs: Number of linear warm-up epochs before the main
            schedule begins.
        gradient_clip: Max gradient norm for gradient clipping (0 disables).
        log_interval: Log to TensorBoard every *N* optimiser steps.
        checkpoint_dir: Directory where model checkpoints are saved.
        tensorboard_dir: Root directory for TensorBoard event files.
        device: Compute device string, e.g. ``"cuda"`` or ``"cpu"``.
    """

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "runs"
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Root configuration object composing all sub-configs.

    Instantiate directly for unit tests or use :func:`load_config` to
    populate from a YAML file with optional CLI overrides.
    """

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    background_cnn: BackgroundCNNConfig = field(default_factory=BackgroundCNNConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    cross_modal: CrossModalConfig = field(default_factory=CrossModalConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path,
    overrides: Optional[List[str]] = None,
) -> Config:
    """Load and validate a :class:`Config` from a YAML file.

    Merging order (later entries take precedence):
      1. Dataclass defaults (``Config`` and its sub-configs).
      2. Values from ``config_path`` YAML file.
      3. Dot-list *overrides* (e.g. ``["training.epochs=50", "gnn.latent_dim=128"]``).

    After merging, a consistency check asserts that
    ``data.saxs_dim + data.waxs_dim == data.total_dim``.

    Args:
        config_path: Path to a YAML configuration file.
        overrides: Optional list of OmegaConf dot-list override strings.

    Returns:
        Fully merged and validated :class:`Config` instance.

    Raises:
        AssertionError: If ``saxs_dim + waxs_dim != total_dim``.
        omegaconf.MissingMandatoryValue: If a required field is absent.
    """
    # Start from the structured (typed) defaults
    schema = OmegaConf.structured(Config)

    # Overlay the user-provided YAML
    file_cfg = OmegaConf.load(config_path)
    merged = OmegaConf.merge(schema, file_cfg)

    # Apply any CLI-level overrides last
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    # Cast to the typed Config dataclass tree
    cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]

    # Sanity check: ensure the split dimensions are consistent
    assert cfg.data.saxs_dim + cfg.data.waxs_dim == cfg.data.total_dim, (
        f"Config inconsistency: saxs_dim ({cfg.data.saxs_dim}) + "
        f"waxs_dim ({cfg.data.waxs_dim}) != total_dim ({cfg.data.total_dim})."
    )

    logger.info("Config loaded from %s", config_path)
    return cfg
