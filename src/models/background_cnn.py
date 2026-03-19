"""1-D CNN background estimator: f_bg = CNN(x_signal, x_ball)."""
from __future__ import annotations
from typing import List
import torch, torch.nn as nn, torch.nn.functional as F

class ResBlock1d(nn.Module):
    def __init__(self, ch, ks, dropout=0.1):
        super().__init__()
        p = ks // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, ks, padding=p, bias=False), nn.BatchNorm1d(ch), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, ks, padding=p, bias=False), nn.BatchNorm1d(ch))
    def forward(self, x): return F.relu(self.net(x) + x)

class BackgroundCNN(nn.Module):
    """Input: (B, L) signal + (B, L) ball  →  Output: (B, L) f_bg ≥ 0."""
    def __init__(self, signal_len, hidden_channels=(64,128,64), kernel_size=5, dropout=0.1):
        super().__init__()
        p = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(2, hidden_channels[0], kernel_size, padding=p, bias=False),
            nn.BatchNorm1d(hidden_channels[0]), nn.ReLU(True))
        layers, in_ch = [], hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size, padding=p, bias=False),
                       nn.BatchNorm1d(out_ch), nn.ReLU(True),
                       ResBlock1d(out_ch, kernel_size, dropout)]
            in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_ch, 1, 1)

    def forward(self, x_signal, x_ball):
        h = self.stem(torch.stack([x_signal, x_ball], dim=1))
        return F.relu(self.head(self.body(h)).squeeze(1))
