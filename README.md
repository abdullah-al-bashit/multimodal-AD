# multimodal_xray_gnn

Multi-modal X-ray scattering analysis framework combining SAXS (300-dim) and WAXS
(290-dim) signals with GATv2-based GNN encoders, rolling-ball background subtraction,
cross-modal attention fusion, and InfoNCE contrastive learning.

## Architecture

```
x ∈ R^590  →  split  →  xs (300) + xw (290)
                          ↓             ↓
                     BackgroundCNN  BackgroundCNN   →  f_bg_s, f_bg_w
                          ↓             ↓
                     xdiff_s = xs - f_bg_s
                     xdiff_w = xw - f_bg_w
                          ↓             ↓
                      GNN_S (PyG)   GNN_W (PyG)     →  z_S, z_W ∈ R^D
                          ↓             ↓
                       CrossModalFusion (Z_mp)       →  L_NCE
                          ↓             ↓
                      Decoder_S    Decoder_W         →  L_MSE_S, L_MSE_W·G

L_Total = ω1·L_BG + ω2·L_MSE_S + ω3·(L_MSE_W·G) + ω4·L_NCE
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install base deps
pip install -r requirements.txt

# 3. Install PyG (match your CUDA — see https://pyg.org/whl/)
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 4. Install project
pip install -e .

# 5. Run tests
pytest tests/ -v

# 6. Train  (place your data at data/raw/spectra.h5 first)
python scripts/train.py --data data/raw/spectra.h5

# 7. Evaluate
python scripts/evaluate.py --data data/raw/spectra.h5 \
    --ckpt checkpoints/checkpoint_best.pt --plot

# 8. TensorBoard
tensorboard --logdir runs/
```

## Data Format

HDF5 file with keys:
- `"spectra"` → `(N, 590)` float32
- `"labels"`  → `(N,)` int  *(optional)*

Or `.npy` array of shape `(N, 590)`.

## Key Config (configs/default.yaml)

| Key | Default | Description |
|-----|---------|-------------|
| `gnn.k_neighbors` | 8 | k-NN graph connectivity |
| `gnn.latent_dim` | 64 | Embedding dimension D |
| `gnn.heads` | 4 | GATv2 attention heads |
| `loss.waxs_peak_positions` | [50,150] | Prior peak indices in WAXS |
| `cross_modal.temperature` | 0.07 | InfoNCE temperature |
