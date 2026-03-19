"""
Dummy scattering data generator for multimodal_xray_gnn.

Simulates realistic SAXS + WAXS spectra for 3 tissue classes:
    0 = Healthy
    1 = Mild Cognitive Impairment (MCI)
    2 = Alzheimer's Disease (AD)

Each class has physically motivated signal characteristics:
    - SAXS: myelin peak intensity decreases with disease severity
    - WAXS: amyloid peaks at positions 50 and 150 grow with disease severity

Output: data/raw/spectra.h5
    "spectra" → (N, 590) float32
    "labels"  → (N,)     int64
"""

import numpy as np
import h5py
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
N_PER_CLASS = 200        # samples per class  → 600 total
SAXS_DIM    = 300
WAXS_DIM    = 290
TOTAL_DIM   = 590
SEED        = 42
OUTPUT_PATH = Path("data/raw/spectra.h5")

rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Signal building blocks
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_peak(length, center, width, amplitude):
    """Add a Gaussian peak to a spectrum."""
    x = np.arange(length)
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

def power_law_background(length, scale=10.0, exponent=-2.0):
    """Simulate the power-law SAXS background (common in real data)."""
    q = np.linspace(0.01, 1.0, length)
    return scale * np.power(q, exponent)

def smooth_noise(length, scale=0.05, smoothing=10):
    """Correlated (smooth) noise — more realistic than white noise."""
    from scipy.ndimage import gaussian_filter1d
    raw = rng.normal(0, scale, length)
    return gaussian_filter1d(raw, sigma=smoothing)


# ─────────────────────────────────────────────────────────────────────────────
# SAXS spectrum generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_saxs(n_samples, disease_severity=0.0):
    """
    Generate SAXS spectra (300 dims).

    disease_severity:
        0.0 = healthy  (strong myelin peak)
        0.5 = MCI      (weakened myelin)
        1.0 = AD       (myelin peak degraded)
    """
    spectra = np.zeros((n_samples, SAXS_DIM), dtype=np.float32)

    for i in range(n_samples):
        # Power-law background (universal in SAXS)
        bg = power_law_background(SAXS_DIM, scale=8.0, exponent=-1.8)

        # Myelin peak at q~0.1 nm-1 → channel ~30
        # Degrades with disease severity
        myelin_amp = rng.uniform(3.0, 5.0) * (1.0 - 0.7 * disease_severity)
        myelin     = gaussian_peak(SAXS_DIM, center=30,  width=8,  amplitude=myelin_amp)

        # Myelin second order peak → channel ~60
        myelin2    = gaussian_peak(SAXS_DIM, center=60,  width=5,  amplitude=myelin_amp * 0.4)

        # Collagen peak → channel ~100 (relatively stable)
        collagen_amp = rng.uniform(1.0, 2.0)
        collagen   = gaussian_peak(SAXS_DIM, center=100, width=12, amplitude=collagen_amp)

        # Void/porosity signal increases with AD
        void_amp   = rng.uniform(0.1, 0.3) * (1.0 + 2.0 * disease_severity)
        void_sig   = gaussian_peak(SAXS_DIM, center=15,  width=20, amplitude=void_amp)

        # Smooth correlated noise
        noise = smooth_noise(SAXS_DIM, scale=0.08)

        spectra[i] = bg + myelin + myelin2 + collagen + void_sig + noise

    # Clip to non-negative (intensities can't be negative)
    return np.clip(spectra, 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# WAXS spectrum generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_waxs(n_samples, disease_severity=0.0):
    """
    Generate WAXS spectra (290 dims).

    Key peaks:
        Position  50 → 4.7 Å spacing → amyloid β-sheet cross peak  (grows with AD)
        Position 150 → 10 Å spacing  → β-sheet stacking distance    (grows with AD)
        Position  80 → lipid chain packing (stable)
        Position 200 → water/hydration peak (stable)
    """
    spectra = np.zeros((n_samples, WAXS_DIM), dtype=np.float32)

    for i in range(n_samples):
        # Flat-ish background with slight slope
        bg = np.linspace(2.0, 0.5, WAXS_DIM) + rng.uniform(0.1, 0.3)

        # ── Amyloid β-sheet peaks (the Alzheimer's signature) ──────────────
        # Position 50  → 4.7 Å  (cross-β spacing — your prior weight!)
        amyloid_4A_amp = rng.uniform(0.05, 0.2) + 2.5 * disease_severity
        amyloid_4A     = gaussian_peak(WAXS_DIM, center=50,  width=4, amplitude=amyloid_4A_amp)

        # Position 150 → 10 Å  (inter-sheet stacking — your prior weight!)
        amyloid_10A_amp = rng.uniform(0.05, 0.15) + 1.8 * disease_severity
        amyloid_10A     = gaussian_peak(WAXS_DIM, center=150, width=6, amplitude=amyloid_10A_amp)

        # ── Stable structural peaks ─────────────────────────────────────────
        # Lipid chain packing → position 80
        lipid_amp = rng.uniform(1.5, 2.5)
        lipid     = gaussian_peak(WAXS_DIM, center=80,  width=5, amplitude=lipid_amp)

        # Myelin lipid bilayer → position 120 (slightly reduced with AD)
        myelin_lipid_amp = rng.uniform(0.8, 1.5) * (1.0 - 0.4 * disease_severity)
        myelin_lipid     = gaussian_peak(WAXS_DIM, center=120, width=7, amplitude=myelin_lipid_amp)

        # Water peak → position 200 (universal, stable)
        water_amp = rng.uniform(3.0, 4.0)
        water     = gaussian_peak(WAXS_DIM, center=200, width=10, amplitude=water_amp)

        # Protein amide → position 240
        protein_amp = rng.uniform(0.5, 1.0)
        protein     = gaussian_peak(WAXS_DIM, center=240, width=8, amplitude=protein_amp)

        # Smooth correlated noise
        noise = smooth_noise(WAXS_DIM, scale=0.06)

        spectra[i] = bg + amyloid_4A + amyloid_10A + lipid + myelin_lipid + water + protein + noise

    return np.clip(spectra, 0, None)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset():
    print("Generating dummy scattering dataset...")
    print(f"  Classes : Healthy(0), MCI(1), AD(2)")
    print(f"  Samples : {N_PER_CLASS} per class = {N_PER_CLASS * 3} total")
    print(f"  Dims    : SAXS={SAXS_DIM}  WAXS={WAXS_DIM}  Total={TOTAL_DIM}")

    # Disease severity per class
    classes = [
        (0, "Healthy", 0.0),
        (1, "MCI",     0.5),
        (2, "AD",      1.0),
    ]

    all_spectra = []
    all_labels  = []

    for label, name, severity in classes:
        print(f"  Generating {N_PER_CLASS} {name} samples (severity={severity})...")
        saxs = generate_saxs(N_PER_CLASS, disease_severity=severity)   # (N, 300)
        waxs = generate_waxs(N_PER_CLASS, disease_severity=severity)   # (N, 290)
        full = np.concatenate([saxs, waxs], axis=1)                    # (N, 590)
        all_spectra.append(full)
        all_labels.append(np.full(N_PER_CLASS, label, dtype=np.int64))

    X      = np.concatenate(all_spectra, axis=0).astype(np.float32)    # (600, 590)
    labels = np.concatenate(all_labels,  axis=0)                       # (600,)

    # Shuffle
    idx = rng.permutation(len(X))
    X, labels = X[idx], labels[idx]

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUTPUT_PATH, "w") as f:
        f.create_dataset("spectra", data=X,      compression="gzip")
        f.create_dataset("labels",  data=labels, compression="gzip")

    print(f"\n✅  Saved → {OUTPUT_PATH}")
    print(f"   spectra shape : {X.shape}")
    print(f"   labels shape  : {labels.shape}")
    print(f"   label counts  : { {i: (labels==i).sum() for i in range(3)} }")
    return X, labels


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check plot (optional)
# ─────────────────────────────────────────────────────────────────────────────

def plot_samples(X, labels):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    class_names = {0: "Healthy", 1: "MCI", 2: "Alzheimer's"}
    colors      = {0: "steelblue", 1: "darkorange", 2: "crimson"}

    for cls in range(3):
        idx    = np.where(labels == cls)[0][0]   # first sample of each class
        sample = X[idx]
        saxs   = sample[:300]
        waxs   = sample[300:]

        # SAXS plot
        ax = axes[0, cls]
        ax.plot(saxs, color=colors[cls], lw=1.2)
        ax.axvline(30,  color="gray", ls="--", lw=0.8, label="Myelin")
        ax.axvline(100, color="green", ls="--", lw=0.8, label="Collagen")
        ax.set_title(f"SAXS — {class_names[cls]}", fontweight="bold")
        ax.set_xlabel("Channel (q index)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # WAXS plot
        ax = axes[1, cls]
        ax.plot(waxs, color=colors[cls], lw=1.2)
        ax.axvline(50,  color="red",    ls="--", lw=0.8, label="Amyloid 4.7Å (pos 50)")
        ax.axvline(150, color="purple", ls="--", lw=0.8, label="β-sheet 10Å (pos 150)")
        ax.axvline(80,  color="gray",   ls=":",  lw=0.8, label="Lipid")
        ax.set_title(f"WAXS — {class_names[cls]}", fontweight="bold")
        ax.set_xlabel("Channel (q index)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "Simulated Scattering Spectra\n"
        "Red dashed = amyloid peaks (grow with AD severity)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("data/raw/sample_spectra.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved → data/raw/sample_spectra.png")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, labels = generate_dataset()

    # Ask user if they want to plot
    try:
        import matplotlib
        plot_samples(X, labels)
    except ImportError:
        print("matplotlib not available — skipping plot")
