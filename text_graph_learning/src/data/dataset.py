"""
Dataset Module — Data Loading and Dataset Classes
===================================================

Purpose:
    Load raw graph datasets and convert them into PyTorch Geometric (PyG)
    Data objects ready for model training.

Responsibilities:
    - Load datasets from data/raw/ (Cora, PubMed, ogbn-arxiv, etc.)
    - Construct graph structure (edge_index, node features)
    - Generate train/validation/test splits
    - Provide DataLoaders for both transductive and inductive settings

Future Implementation Plan:
    - load_dataset():  Main entry point — returns PyG Data/DataLoader
    - CoraDataset:     Citation network with paper text attributes
    - PubMedDataset:   Larger citation network
    - OGBNDataset:      Open Graph Benchmark datasets

Usage (future):
    from src.data.dataset import load_dataset
    data = load_dataset(config)
"""

from typing import Any, Dict, Optional

# TODO: Import torch and torch_geometric when implementing
# import torch
# from torch_geometric.data import Data, DataLoader


def load_dataset(config: Dict[str, Any]):
    """
    Load a graph dataset based on configuration.

    Args:
        config: Configuration dictionary (from YAML).
            Expected keys:
                dataset.name:   Dataset identifier (e.g., "cora")
                dataset.path:   Path to raw data directory

    Returns:
        PyG Data object (or DataLoader for inductive models).

    TODO:
        - Implement dataset registry (name → loader mapping)
        - Support train/val/test split generation
        - Support both transductive and inductive settings
    """
    raise NotImplementedError("Dataset loading will be implemented in Phase 2.")


class GraphDataset:
    """
    Base class for graph datasets.

    TODO:
        - Define __init__, __len__, __getitem__
        - Integrate with PyG Data objects
        - Support feature and label extraction
    """

    def __init__(self, name: str, root: str):
        self.name = name
        self.root = root
        # TODO: Load raw data, build graph, extract features
        raise NotImplementedError

    def get_splits(self, train_ratio=0.6, val_ratio=0.2):
        """
        Generate train/validation/test splits.

        TODO:
            - Support random, stratified, and fixed splits
            - Return masks for transductive setting
        """
        raise NotImplementedError
