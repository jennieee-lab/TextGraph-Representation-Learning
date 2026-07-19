"""
Models Module — Text-enhanced Graph Representation Learning
===========================================================

Model registry and factory:
  - Baselines: MLP, Node2Vec
  - Graph Neural Networks: GCN, GraphSAGE, GAT
  - Text Encoder: BERT-based encoder (future)
  - Fusion: TextGraphFusion (future)
"""

from typing import Any, Dict

from src.models.mlp import MLP
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE


_MODEL_REGISTRY = {
    "mlp": MLP,
    "gcn": GCN,
    "graphsage": GraphSAGE,
}


def build_model(config: Dict[str, Any], num_features: int, num_classes: int):
    """Construct a model from configuration.

    Args:
        config: Configuration dict. Key model.name selects the architecture.
        num_features: Input feature dimension (from dataset).
        num_classes: Number of target classes (from dataset).

    Returns:
        Initialized model instance.
    """
    name = config["model"]["name"].lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    model_cls = _MODEL_REGISTRY[name]
    model = model_cls(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=config["model"].get("hidden_dimension", 128),
        num_layers=config["model"].get("num_layers", 2),
        dropout=config["model"].get("dropout", 0.5),
    )
    return model


__all__ = ["build_model", "MLP", "GCN", "GraphSAGE"]
