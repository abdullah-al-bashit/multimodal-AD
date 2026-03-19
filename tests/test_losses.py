import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.losses.losses import background_loss, info_nce_loss, TotalLoss

B,D = 4,64

def test_bg_loss_positive():
    assert background_loss(torch.abs(torch.randn(B,300)),
                           torch.abs(torch.randn(B,300))).item() >= 0

def test_nce_scalar():
    l = info_nce_loss(torch.randn(B,D), torch.randn(B,D))
    assert l.dim()==0

def test_total_loss():
    bd = TotalLoss()(
        torch.abs(torch.randn(B,300)), torch.abs(torch.randn(B,290)),
        torch.abs(torch.randn(B,300)), torch.abs(torch.randn(B,290)),
        torch.abs(torch.randn(B,300)), torch.abs(torch.randn(B,290)),
        torch.randn(B,300), torch.randn(B,290),
        torch.randn(B,D),   torch.randn(B,D))
    assert bd.total.item() >= 0
