"""
GraphSAGE Model — Inductive Representation Learning on Large Graphs
=====================================================================

Implements GraphSAGE (Hamilton et al., 2017) with mean aggregation.
Supports inductive learning — can generalize to unseen nodes.

Reference:
    Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation
    Learning on Large Graphs. NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """GraphSAGE for node classification with mean aggregator.

    Samples and aggregates features from a node's local neighborhood.
    Unlike GCN, GraphSAGE is inductive — it can generate embeddings for
    nodes not seen during training.

    Args:
        num_features: Dimension of input node features.
        num_classes: Number of target classes.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of SAGE layers (default 2).
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
        self.name = "GraphSAGE"

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_classes))

        self.dropout = dropout

    def forward(self, x, edge_index):
        """Forward pass: neighborhood sampling and mean aggregation.

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


__all__ = ["GraphSAGE"]
