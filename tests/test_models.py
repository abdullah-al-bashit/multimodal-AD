import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest, torch
from src.models.background_cnn   import BackgroundCNN
from src.models.decoder          import Decoder
from src.models.cross_modal      import CrossModalFusion
from src.models.scattering_model import ScatteringModel

B = 4

def test_bg_cnn_shape():
    cnn = BackgroundCNN(300)
    out = cnn(torch.randn(B,300), torch.randn(B,300))
    assert out.shape==(B,300) and (out>=0).all()

@pytest.mark.parametrize("d",[300,290])
def test_decoder(d):
    out = Decoder(64,d)(torch.randn(B,64))
    assert out.shape==(B,d) and (out>=0).all()

@pytest.mark.parametrize("m",["attention","concat","add"])
def test_fusion(m):
    assert CrossModalFusion(64,m)(torch.randn(B,64),torch.randn(B,64)).shape==(B,64)

def test_full_model():
    m = ScatteringModel(300,290,gnn_hidden=32,gnn_latent=16,gnn_layers=2,gnn_k=4)
    out = m(torch.randn(B,300),torch.randn(B,290),
            torch.randn(B,300),torch.randn(B,290))
    assert out.z_mp.shape==(B,16)
    assert out.xhat_s.shape==(B,300)
    assert out.xhat_w.shape==(B,290)
