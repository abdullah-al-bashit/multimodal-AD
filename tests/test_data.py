import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pytest
from src.data.preprocessing import split_spectrum, rolling_ball_background, ScatteringPreprocessor

def test_split_shapes():
    xs, xw = split_spectrum(np.random.rand(10,590))
    assert xs.shape==(10,300) and xw.shape==(10,290)

def test_split_wrong_dim():
    with pytest.raises(ValueError): split_spectrum(np.random.rand(10,500))

def test_rolling_ball_shape():
    x = np.random.rand(8,300)
    assert rolling_ball_background(x,20).shape == x.shape

def test_rolling_ball_nonneg():
    x = np.abs(np.random.rand(4,300))
    assert (rolling_ball_background(x,10)>=0).all()

def test_preprocessor():
    X = np.random.rand(50,590).astype(np.float32)
    xs,xw,xbs,xbw = ScatteringPreprocessor().fit_transform(X)
    assert xs.shape==(50,300) and xw.shape==(50,290)
