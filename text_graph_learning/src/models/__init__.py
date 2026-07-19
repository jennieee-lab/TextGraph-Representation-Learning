"""
Models Module — Text-enhanced Graph Representation Learning
===========================================================

This module contains all model definitions:
  - Baselines: MLP, Node2Vec
  - Graph Neural Networks: GCN, GraphSAGE, GAT
  - Text Encoder: BERT-based encoder
  - Fusion: TextGraphFusion (combines text + graph embeddings)

Usage (future):
    from src.models import build_model
    model = build_model(config)
"""

from typing import Any, Dict


def build_model(config: Dict[str, Any]):
    """
    Model factory — construct model from configuration.

    Args:
        config: Configuration dictionary. Key model.name determines the model.

    TODO:
        - Map model name string to model class
        - Return initialized model
    """
    raise NotImplementedError("Model construction will be implemented in Phase 3-5.")


__all__ = ["build_model"]
