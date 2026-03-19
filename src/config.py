"""Centralised configuration management using OmegaConf."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    saxs_dim: int = 300
    waxs_dim: int = 290
    total_dim: int = 590
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42

@dataclass
class PreprocessingConfig:
    ball_radius: int = 20
    normalize: bool = True
    norm_mode: str = "zscore"

@dataclass
class BackgroundCNNConfig:
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 64])
    kernel_size: int = 5
    dropout: float = 0.1

@dataclass
class GNNConfig:
    k_neighbors: int = 8
    hidden_dim: int = 128
    latent_dim: int = 64
    num_layers: int = 3
    heads: int = 4
    dropout: float = 0.1

@dataclass
class CrossModalConfig:
    method: str = "attention"
    temperature: float = 0.07

@dataclass
class DecoderConfig:
    hidden_dim: int = 128
    dropout: float = 0.1

@dataclass
class LossConfig:
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

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    background_cnn: BackgroundCNNConfig = field(default_factory=BackgroundCNNConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    cross_modal: CrossModalConfig = field(default_factory=CrossModalConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

def load_config(config_path: str | Path, overrides: Optional[List[str]] = None) -> Config:
    schema  = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(config_path)
    merged  = OmegaConf.merge(schema, file_cfg)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    cfg: Config = OmegaConf.to_object(merged)  # type: ignore[assignment]
    assert cfg.data.saxs_dim + cfg.data.waxs_dim == cfg.data.total_dim
    logger.info("Config loaded from %s", config_path)
    return cfg
