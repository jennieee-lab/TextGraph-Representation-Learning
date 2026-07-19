"""
GCN Model — Graph Convolutional Network
=========================================

Implements the Graph Convolutional Network from Kipf & Welling (2017).
Uses spectral graph convolution via the Chebyshev approximation (1st-order).

Reference:
    Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification
    with Graph Convolutional Networks. ICLR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Graph Convolutional Network for node classification.

    Propagates messages along edges using spectral graph convolution.
    Each layer aggregates features from the node's 1-hop neighborhood.

    Args:
        num_features: Dimension of input node features.
        num_classes: Number of target classes.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GCN layers (default 2).
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
        self.name = "GCN"

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))

        self.dropout = dropout

    def forward(self, x, edge_index):
        """Forward pass: layer-wise message passing along edges.

        Args:
            x: Node feature matrix [num_nodes, num_features].
            edge_index: Edge index [2, num_edges].

        Returns:
            Logits [num_nodes, num_classes].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


__all__ = ["GCN"]
