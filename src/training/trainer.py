"""
trainer.py
==========
Training loop with AMP, gradient clipping, LR warm-up, cosine scheduling,
TensorBoard logging, and checkpoint management.

Design decisions
----------------
* **Automatic Mixed Precision (AMP)**: ``torch.cuda.amp.GradScaler`` is used
  on CUDA devices to reduce memory usage and speed up training with float16
  arithmetic while keeping float32 master weights.
* **Gradient clipping**: ``nn.utils.clip_grad_norm_`` prevents exploding
  gradients, which are common in GNN training.
* **LR warm-up**: A linear warm-up over the first ``warmup_epochs`` avoids
  large gradient updates with a cold model early in training.
* **Scheduler**: Cosine annealing (default) decays the LR smoothly to
  ``1e-6``; a step-decay variant is also supported.
* **Checkpointing**: The best validation checkpoint and periodic epoch
  snapshots are saved to ``cfg.training.checkpoint_dir``.
* **TensorBoard**: Per-step and per-epoch scalars are written to
  ``cfg.training.tensorboard_dir``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import Config
from ..losses.losses import LossBreakdown, TotalLoss
from ..models.scattering_model import ScatteringModel

logger = logging.getLogger(__name__)

# Interval (in epochs) at which periodic checkpoints are saved
_CHECKPOINT_EPOCH_INTERVAL = 10


class Trainer:
    """Manages the full training lifecycle for :class:`ScatteringModel`.

    Responsibilities:

    * Moving model and criterion to the target device.
    * Constructing the AdamW optimiser and LR scheduler.
    * Running train / validation epochs with AMP support.
    * Writing scalars to TensorBoard.
    * Saving best-val and periodic epoch checkpoints.
    * Resuming from a checkpoint via :meth:`fit`.

    Args:
        model: The :class:`~src.models.scattering_model.ScatteringModel`
            to train.
        criterion: The :class:`~src.losses.losses.TotalLoss` module.
        cfg: Full :class:`~src.config.Config` configuration object.

    Example::

        trainer = Trainer(model, criterion, cfg)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: ScatteringModel,
        criterion: TotalLoss,
        cfg: Config,
    ) -> None:
        self.cfg = cfg

        # Resolve device: fall back to CPU if CUDA was requested but unavailable
        requested_device = cfg.training.device
        self.device = torch.device(
            requested_device
            if (torch.cuda.is_available() or requested_device == "cpu")
            else "cpu"
        )
        if self.device.type == "cpu" and requested_device != "cpu":
            logger.warning(
                "Requested device '%s' is unavailable; falling back to CPU.",
                requested_device,
            )

        # Move model and loss function to the training device
        self.model     = model.to(self.device)
        self.criterion = criterion.to(self.device)

        # AdamW: decoupled weight decay (better than L2 regularisation with Adam)
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        self.scheduler = self._build_scheduler()

        # GradScaler is a no-op on CPU (enabled=False)
        self.scaler = GradScaler(enabled=self.device.type == "cuda")

        # I/O setup
        self.ckpt_dir = Path(cfg.training.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(cfg.training.tensorboard_dir)

        # State tracking
        self._global_step: int = 0          # total optimiser steps across all epochs
        self._best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Scheduler construction
    # ------------------------------------------------------------------

    def _build_scheduler(self) -> Optional[object]:
        """Instantiate the LR scheduler specified in config.

        Returns:
            A PyTorch LR scheduler instance, or ``None`` if the scheduler
            name is not recognised.
        """
        sched_name = self.cfg.training.lr_scheduler
        total_epochs = self.cfg.training.epochs

        if sched_name == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=1e-6
            )
        if sched_name == "step":
            return StepLR(
                self.optimizer, step_size=total_epochs // 3, gamma=0.1
            )

        logger.warning("Unknown lr_scheduler '%s'; no scheduler will be used.", sched_name)
        return None

    # ------------------------------------------------------------------
    # Linear warm-up
    # ------------------------------------------------------------------

    def _apply_warmup(self, epoch: int) -> None:
        """Linearly scale the LR from near-zero to the configured initial value.

        Only active for ``epoch < warmup_epochs``.  After warm-up, the main
        scheduler takes over.

        Args:
            epoch: Current epoch index (0-based).
        """
        warmup_epochs = self.cfg.training.warmup_epochs
        if epoch < warmup_epochs:
            # Scale: epoch=0 → LR≈0, epoch=warmup_epochs-1 → LR=learning_rate
            warmup_lr = self.cfg.training.learning_rate * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr

    # ------------------------------------------------------------------
    # Single batch step
    # ------------------------------------------------------------------

    def _step(self, batch: Dict[str, torch.Tensor], is_train: bool) -> LossBreakdown:
        """Process one mini-batch: forward pass, optional backward + update.

        AMP context is used during training on CUDA; a plain
        ``torch.no_grad()`` context is used during validation.

        Args:
            batch: Dictionary produced by :class:`~src.data.dataset.ScatteringDataset`.
            is_train: If ``True``, perform the backward pass and optimiser step.

        Returns:
            :class:`~src.losses.losses.LossBreakdown` with all loss components.
        """
        # Move batch tensors to the training device
        xs     = batch["xs"].to(self.device)
        xw     = batch["xw"].to(self.device)
        xball_s = batch["xball_s"].to(self.device)
        xball_w = batch["xball_w"].to(self.device)

        # Select appropriate computation context
        if is_train:
            # AMP autocast on CUDA; standard float32 on CPU
            ctx = (
                autocast(device_type=self.device.type)
                if self.device.type == "cuda"
                else torch.enable_grad()
            )
        else:
            ctx = torch.no_grad()

        with ctx:
            out = self.model(xs, xw, xball_s, xball_w)
            loss_breakdown = self.criterion(
                out.f_bg_s, out.f_bg_w,
                xball_s, xball_w,
                out.xhat_s, out.xhat_w,
                out.xdiff_s, out.xdiff_w,
                out.z_s, out.z_w,
            )

        if is_train:
            self.optimizer.zero_grad(set_to_none=True)  # more efficient than zero_grad()

            # Scale the loss to prevent underflow in float16 gradients
            self.scaler.scale(loss_breakdown.total).backward()

            # Unscale before gradient clipping so the clip threshold is in
            # the original (unscaled) gradient space
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.training.gradient_clip
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._global_step += 1

            # Periodic TensorBoard logging
            if self._global_step % self.cfg.training.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("train/loss", loss_breakdown.total.item(), self._global_step)
                self.writer.add_scalar("train/lr",   current_lr,                 self._global_step)

        return loss_breakdown

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        is_train: bool,
        epoch: int,
    ) -> Dict[str, float]:
        """Iterate over all batches in *loader* and aggregate loss metrics.

        Args:
            loader: DataLoader for the current split.
            is_train: ``True`` for the training loop, ``False`` for validation.
            epoch: Current epoch index (used for logging context).

        Returns:
            Dictionary with keys ``total``, ``bg``, ``mse_s``, ``mse_w``,
            ``nce``, and ``t`` (wall-clock seconds for the epoch).
        """
        self.model.train(is_train)

        # Accumulate loss sums over all batches
        totals: Dict[str, float] = dict(total=0.0, bg=0.0, mse_s=0.0, mse_w=0.0, nce=0.0)
        t_start = time.perf_counter()

        for batch in loader:
            breakdown = self._step(batch, is_train)
            for key in totals:
                totals[key] += getattr(breakdown, key).item()

        num_batches = len(loader)
        means = {key: val / num_batches for key, val in totals.items()}
        means["t"] = time.perf_counter() - t_start
        return means

    # ------------------------------------------------------------------
    # Public training entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_path: Optional[str | Path] = None,
    ) -> None:
        """Run the full training loop.

        If *resume_path* is provided, the model/optimiser/scaler states and
        the starting epoch are restored from that checkpoint before training.

        Args:
            train_loader: DataLoader for the training split.
            val_loader:   DataLoader for the validation split.
            resume_path:  Optional path to a ``.pt`` checkpoint to resume from.
        """
        start_epoch = 0
        if resume_path is not None:
            start_epoch = self._load_checkpoint(resume_path)
            logger.info("Resuming from '%s' at epoch %d.", resume_path, start_epoch)

        for epoch in range(start_epoch, self.cfg.training.epochs):
            # Apply linear LR warm-up (no-op once warm-up is complete)
            self._apply_warmup(epoch)

            # Train and validate
            train_metrics = self._run_epoch(train_loader, is_train=True,  epoch=epoch)
            val_metrics   = self._run_epoch(val_loader,   is_train=False, epoch=epoch)

            # Advance the LR scheduler (skip during warm-up epochs)
            if self.scheduler is not None and epoch >= self.cfg.training.warmup_epochs:
                self.scheduler.step()

            # Log per-epoch scalars to TensorBoard
            for key, val in train_metrics.items():
                if key != "t":
                    self.writer.add_scalar(f"epoch/train_{key}", val, epoch)
            for key, val in val_metrics.items():
                if key != "t":
                    self.writer.add_scalar(f"epoch/val_{key}", val, epoch)

            # Console summary
            logger.info(
                "Epoch %03d | train=%.4f  val=%.4f  "
                "[bg=%.4f  mse_s=%.4f  mse_w=%.4f  nce=%.4f]  %.1fs",
                epoch + 1,
                train_metrics["total"], val_metrics["total"],
                val_metrics["bg"], val_metrics["mse_s"],
                val_metrics["mse_w"], val_metrics["nce"],
                val_metrics["t"],
            )

            # Save the best-validation checkpoint
            if val_metrics["total"] < self._best_val_loss:
                self._best_val_loss = val_metrics["total"]
                self._save_checkpoint(epoch, tag="best")
                logger.info("  New best validation loss: %.4f", self._best_val_loss)

            # Periodic checkpoint every N epochs
            if (epoch + 1) % _CHECKPOINT_EPOCH_INTERVAL == 0:
                self._save_checkpoint(epoch, tag=f"epoch_{epoch + 1:04d}")

        self.writer.close()
        logger.info("Training complete.  Best val loss: %.4f", self._best_val_loss)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, tag: str) -> None:
        """Serialise model, optimiser, and scaler state to a ``.pt`` file.

        Args:
            epoch: Current epoch index (0-based); stored as ``epoch + 1`` so
                   the saved value reflects the number of completed epochs.
            tag:   String tag appended to the filename (e.g. ``"best"``).
        """
        checkpoint = {
            "epoch":           epoch + 1,            # number of completed epochs
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "best_val_loss":   self._best_val_loss,
        }
        save_path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, save_path)
        logger.debug("Checkpoint saved → %s", save_path)

    def _load_checkpoint(self, path: str | Path) -> int:
        """Restore model, optimiser, and scaler state from a checkpoint file.

        Args:
            path: Path to a ``.pt`` checkpoint previously saved by
                  :meth:`_save_checkpoint`.

        Returns:
            The epoch index to resume from (i.e. the next epoch to run).
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        # "epoch" stores the number of completed epochs → that is the next start
        return checkpoint.get("epoch", 0)
