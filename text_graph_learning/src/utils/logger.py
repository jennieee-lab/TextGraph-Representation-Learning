"""
Logger Module — Logging Configuration
======================================

Provide a unified logging interface for experiments.
Logs are written to both console and a file in the experiment output directory.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name:    Logger name (typically __name__).
        log_dir: Directory for log files. If None, only console output.
        level:   Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "experiment.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


__all__ = ["get_logger"]
