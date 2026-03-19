from setuptools import setup, find_packages

setup(
    name="multimodal_xray_gnn",
    version="0.1.0",
    author="Ganesan",
    description=(
        "Multi-modal X-ray scattering analysis framework (SAXS + WAXS) "
        "with GATv2-based GNN encoders, rolling-ball background subtraction, "
        "cross-modal attention fusion, and InfoNCE contrastive learning."
    ),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "h5py>=3.9.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.14.0",
    ],
)
