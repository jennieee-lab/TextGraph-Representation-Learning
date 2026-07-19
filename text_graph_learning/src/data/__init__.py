"""
Data Module — Text-enhanced Graph Representation Learning
=========================================================

This module handles:
  1. Dataset loading (raw data → PyG Data objects)
  2. Data preprocessing (text cleaning, graph construction, feature extraction)

Submodules:
  - dataset.py:       Dataset classes and data loaders
  - preprocessing.py:  Text cleaning, graph construction, feature engineering
"""

from .dataset import load_dataset
from .preprocessing import preprocess_data

__all__ = ["load_dataset", "preprocess_data"]
