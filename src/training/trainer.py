"""
Trainer: AMP + grad-clip + warmup + cosine LR + TensorBoard + checkpointing.
"""
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Dict, Optional
import torch, torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..models.scattering_model import ScatteringModel
from ..losses.losses import TotalLoss, LossBreakdown
from ..config import Config

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: ScatteringModel, criterion: TotalLoss, cfg: Config):
        self.cfg  = cfg
        dev = cfg.training.device
        self.device = torch.device(dev if (torch.cuda.is_available() or dev=="cpu") else "cpu")
        self.model     = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate,
                               weight_decay=cfg.training.weight_decay)
        self.scheduler = self._build_scheduler()
        self.scaler    = GradScaler(enabled=self.device.type=="cuda")
        self.ckpt_dir  = Path(cfg.training.checkpoint_dir); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer    = SummaryWriter(cfg.training.tensorboard_dir)
        self._step_n   = 0
        self._best_val = float("inf")

    def _build_scheduler(self):
        s, e = self.cfg.training.lr_scheduler, self.cfg.training.epochs
        if s == "cosine": return CosineAnnealingLR(self.optimizer, T_max=e, eta_min=1e-6)
        if s == "step":   return StepLR(self.optimizer, step_size=e//3, gamma=0.1)
        return None

    def _warmup(self, epoch):
        w = self.cfg.training.warmup_epochs
        if epoch < w:
            lr = self.cfg.training.learning_rate * (epoch+1) / w
            for pg in self.optimizer.param_groups: pg["lr"] = lr

    def _step(self, batch, train):
        xs, xw = batch["xs"].to(self.device), batch["xw"].to(self.device)
        xbs    = batch["xball_s"].to(self.device)
        xbw    = batch["xball_w"].to(self.device)
        ctx    = (autocast(device_type=self.device.type)
                  if self.device.type=="cuda" else
                  (torch.enable_grad() if train else torch.no_grad()))
        with ctx:
            out = self.model(xs, xw, xbs, xbw)
            bd  = self.criterion(out.f_bg_s, out.f_bg_w, xbs, xbw,
                                 out.xhat_s, out.xhat_w,
                                 out.xdiff_s, out.xdiff_w, out.z_s, out.z_w)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(bd.total).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
            self.scaler.step(self.optimizer); self.scaler.update()
            self._step_n += 1
            if self._step_n % self.cfg.training.log_interval == 0:
                self.writer.add_scalar("train/loss", bd.total.item(), self._step_n)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self._step_n)
        return bd

    def _epoch(self, loader, train, epoch):
        self.model.train(train)
        tot = dict(total=0., bg=0., mse_s=0., mse_w=0., nce=0.)
        t0  = time.perf_counter()
        for batch in loader:
            bd = self._step(batch, train)
            for k in tot: tot[k] += getattr(bd, k).item()
        n = len(loader)
        means = {k: v/n for k,v in tot.items()}
        means["t"] = time.perf_counter() - t0
        return means

    def fit(self, train_loader, val_loader, resume_path=None):
        start = 0
        if resume_path: start = self._load_ckpt(resume_path)
        for ep in range(start, self.cfg.training.epochs):
            self._warmup(ep)
            tr = self._epoch(train_loader, True,  ep)
            vl = self._epoch(val_loader,   False, ep)
            if self.scheduler and ep >= self.cfg.training.warmup_epochs:
                self.scheduler.step()
            for k,v in tr.items():
                if k!="t": self.writer.add_scalar(f"epoch/train_{k}", v, ep)
            for k,v in vl.items():
                if k!="t": self.writer.add_scalar(f"epoch/val_{k}",   v, ep)
            logger.info("Ep %03d | tr=%.4f vl=%.4f bg=%.4f ms=%.4f mw=%.4f nce=%.4f | %.1fs",
                        ep+1, tr["total"], vl["total"], vl["bg"],
                        vl["mse_s"], vl["mse_w"], vl["nce"], vl["t"])
            if vl["total"] < self._best_val:
                self._best_val = vl["total"]
                self._save_ckpt(ep, "best")
                logger.info("  ✓ New best: %.4f", self._best_val)
            if (ep+1) % 10 == 0: self._save_ckpt(ep, f"epoch_{ep+1:04d}")
        self.writer.close()
        logger.info("Done. Best val=%.4f", self._best_val)

    def _save_ckpt(self, epoch, tag):
        torch.save(dict(epoch=epoch+1, model_state=self.model.state_dict(),
                        optimizer_state=self.optimizer.state_dict(),
                        scaler_state=self.scaler.state_dict(),
                        best_val=self._best_val),
                   self.ckpt_dir / f"checkpoint_{tag}.pt")

    def _load_ckpt(self, path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state"])
        self.optimizer.load_state_dict(ck["optimizer_state"])
        self.scaler.load_state_dict(ck["scaler_state"])
        self._best_val = ck.get("best_val", float("inf"))
        return ck.get("epoch", 0)
