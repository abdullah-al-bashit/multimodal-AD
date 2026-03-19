"""Logging configuration."""
from __future__ import annotations
import logging, sys
from pathlib import Path

def setup_logging(level="INFO", log_file=None):
    fmt  = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
    hdlr = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        hdlr.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=hdlr, force=True)
    for lib in ("matplotlib", "PIL", "h5py", "torch_geometric"):
        logging.getLogger(lib).setLevel(logging.WARNING)
