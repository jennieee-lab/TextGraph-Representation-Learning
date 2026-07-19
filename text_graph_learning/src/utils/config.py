"""
Config Module — YAML Configuration Loader
===========================================

Load experiment configurations from YAML files.
Supports simple override via nested key access.
"""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_device(device_str: str = "auto") -> str:
    """Resolve device string to actual device.

    Args:
        device_str: "auto", "cuda", or "cpu".

    Returns:
        Resolved device string.
    """
    if device_str == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_str


__all__ = ["load_config", "get_device"]
