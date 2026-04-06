"""
logging_utils.py
================
Project-wide logging configuration.

Call :func:`setup_logging` once at the entry point of any script or
application before importing project modules.  Subsequent calls to
``logging.getLogger(__name__)`` anywhere in the codebase will then
inherit the configured format and level.

Design
------
* A :class:`~logging.StreamHandler` writing to ``stdout`` is always added.
* An optional :class:`~logging.FileHandler` is added when ``log_file`` is
  provided; the parent directory is created if it does not exist.
* Noisy third-party loggers (matplotlib, PIL, h5py, torch_geometric) are
  silenced to ``WARNING`` level to keep output focused on project logs.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the root logger with a consistent format.

    Should be called once, at the top of each entry-point script, before
    any project modules emit log messages.

    Args:
        level: Root log level as a string – one of ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.  Case-insensitive.
            Defaults to ``"INFO"``.
        log_file: Optional path to a log file.  The parent directory is
            created automatically.  Pass ``None`` (default) to log to
            ``stdout`` only.
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # ``force=True`` removes any previously added handlers on the root logger,
    # ensuring repeated calls in interactive sessions don't duplicate output.
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

    # Suppress verbose output from common third-party libraries
    _noisy_libraries = ("matplotlib", "PIL", "h5py", "torch_geometric")
    for lib_name in _noisy_libraries:
        logging.getLogger(lib_name).setLevel(logging.WARNING)
