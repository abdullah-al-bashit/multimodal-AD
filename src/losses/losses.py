"""
Training losses:
  L_Total = ω1·L_BG + ω2·L_MSE_S + ω3·(L_MSE_W·G) + ω4·L_NCE
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch, torch.nn as nn, torch.nn.functional as F

def background_loss(f_bg, x_ball, lambda1=0.01, lambda2=0.1):
    guide  = F.mse_loss(f_bg, x_ball)
    d2     = f_bg[:, 2:] - 2*f_bg[:, 1:-1] + f_bg[:, :-2]
    smooth = lambda1 * d2.pow(2).mean()
    pos    = lambda2 * F.relu(-f_bg).mean()
    return guide + smooth + pos

def mse_reconstruction_loss(x_hat, x_target):
    return F.mse_loss(x_hat, x_target)

def prior_weighted_mse_loss(x_hat_w, x_target_w, peak_positions, peak_weight=5.0):
    w = torch.ones(x_hat_w.shape[-1], device=x_hat_w.device)
    for p in peak_positions:
        if 0 <= p < len(w): w[p] = peak_weight
    return ((x_hat_w - x_target_w).pow(2) * w.unsqueeze(0)).mean()

def info_nce_loss(z_s, z_w, temperature=0.07):
    B      = z_s.shape[0]
    z_s    = F.normalize(z_s, dim=-1)
    z_w    = F.normalize(z_w, dim=-1)
    logits = torch.mm(z_s, z_w.T) / temperature
    labels = torch.arange(B, device=z_s.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

@dataclass
class LossBreakdown:
    total: torch.Tensor
    bg:    torch.Tensor
    mse_s: torch.Tensor
    mse_w: torch.Tensor
    nce:   torch.Tensor

class TotalLoss(nn.Module):
    def __init__(self, omega1=1.0, omega2=1.0, omega3=1.0, omega4=0.5,
                 lambda1=0.01, lambda2=0.1,
                 waxs_peak_positions=(50,150), waxs_peak_weight=5.0,
                 temperature=0.07):
        super().__init__()
        self.o1, self.o2, self.o3, self.o4 = omega1, omega2, omega3, omega4
        self.l1, self.l2 = lambda1, lambda2
        self.peaks, self.pw = list(waxs_peak_positions), waxs_peak_weight
        self.tau = temperature

    def forward(self, f_bg_s, f_bg_w, xball_s, xball_w,
                xhat_s, xhat_w, xdiff_s, xdiff_w, z_s, z_w) -> LossBreakdown:
        l_bg   = (background_loss(f_bg_s, xball_s, self.l1, self.l2) +
                  background_loss(f_bg_w, xball_w, self.l1, self.l2))
        l_ms   = mse_reconstruction_loss(xhat_s, xdiff_s)
        l_mw   = prior_weighted_mse_loss(xhat_w, xdiff_w, self.peaks, self.pw)
        l_nce  = info_nce_loss(z_s, z_w, self.tau)
        total  = self.o1*l_bg + self.o2*l_ms + self.o3*l_mw + self.o4*l_nce
        return LossBreakdown(total, l_bg.detach(), l_ms.detach(),
                             l_mw.detach(), l_nce.detach())
