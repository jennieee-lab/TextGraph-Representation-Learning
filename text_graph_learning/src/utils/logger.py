"""
Logger Module — Logging Configuration
======================================

Purpose:
    Provide a unified logging interface for experiments.
    Logs are written to both console and a file in the experiment output directory.

Future Implementation Plan:
    - get_logger():  Return a configured logger instance
    - Support TensorBoard writer integration
    - Support W&B logging

Usage (future):
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import sys
from pathlib import Path

# TODO: Integrate with TensorBoard SummaryWriter
# from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger instance.

    Args:
        name:    Logger name (typically __name__).
        log_dir: Directory for log files. If None, only console output.
        level:   Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.

    TODO:
        - Add file handler (write to log_dir/experiment.log)
        - Add TensorBoard writer
        - Support structured logging format
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid duplicate handlers
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # TODO: File handler (if log_dir provided)

    return logger
