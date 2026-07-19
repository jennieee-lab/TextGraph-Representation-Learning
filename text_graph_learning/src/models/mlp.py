"""
MLP Model — Multi-Layer Perceptron Baseline
=============================================

A structural-agnostic baseline that uses only node features (no graph topology).
Establishes a lower bound for graph-aware models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP as PyGMLP


class MLP(nn.Module):
    """Multi-Layer Perceptron for node classification.

    Uses only node features — does not leverage graph structure.
    This establishes a baseline that graph-aware models should beat.

    Args:
        num_features: Dimension of input node features.
        num_classes: Number of target classes.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of MLP layers (default 2).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.name = "MLP"

        layers = []
        in_dim = num_features
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        """Forward pass. edge_index is accepted but ignored (structure-agnostic)."""
        return self.mlp(x)


__all__ = ["MLP"]
