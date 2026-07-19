"""
Seed Module — Reproducibility Utilities
=========================================

Set random seeds across all libraries (Python, NumPy, PyTorch)
to ensure experiment reproducibility.
"""

import random


def set_seed(seed: int = 42):
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["set_seed"]
