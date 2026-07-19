"""
Dataset Module — Data Loading for Graph Datasets
===================================================

Load graph datasets via PyTorch Geometric (PyG).
Currently supports citation networks (Cora, Citeseer, PubMed) through Planetoid.
"""

from typing import Any, Dict

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


_DATASET_REGISTRY = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def load_dataset(config: Dict[str, Any]):
    """Load a graph dataset based on configuration.

    Args:
        config: Configuration dictionary. Expected keys:
            dataset.name: Dataset identifier (e.g., "cora")
            dataset.path: Path to raw data directory

    Returns:
        PyG Data object with graph structure, features, and labels.
    """
    name = config["dataset"]["name"].lower()
    path = config["dataset"].get("path", "data/raw")

    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {list(_DATASET_REGISTRY.keys())}"
        )

    dataset = Planetoid(
        root=path,
        name=_DATASET_REGISTRY[name],
        transform=T.NormalizeFeatures(),
    )

    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.dataset_name = name

    return data


def get_dataset_info(data) -> Dict[str, int]:
    """Return basic statistics about the loaded graph.

    Args:
        data: PyG Data object.

    Returns:
        Dictionary with num_nodes, num_edges, num_features, num_classes.
    """
    return {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": data.num_node_features,
        "num_classes": data.num_classes,
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
    }


__all__ = ["load_dataset", "get_dataset_info"]
