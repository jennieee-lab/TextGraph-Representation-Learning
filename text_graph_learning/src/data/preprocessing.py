"""
Preprocessing Module — Data Preprocessing & Graph Construction
===============================================================

Handles text and graph preprocessing for the text-graph fusion pipeline.

Current status:
  - Graph loading and feature normalization are handled by PyG's Planetoid.
  - Text-specific preprocessing (cleaning, BERT encoding) is prepared for
    the future text-graph fusion phase (Phase 5).

Future Implementation Plan (Phase 5):
  - clean_text():               Text normalization for raw node attributes
  - extract_text_features():    BERT-based text embedding extraction
  - TextGraphFusion integration with GNN structural embeddings
"""

from typing import Any, Dict, List


def preprocess_data(config: Dict[str, Any]):
    """Run the preprocessing pipeline.

    Currently a no-op — PyG's Planetoid handles dataset loading,
    feature normalization, and graph construction automatically.
    Text preprocessing will be added in Phase 5 (text-graph fusion).

    Args:
        config: Configuration dictionary from YAML.
    """
    return {"status": "Preprocessing handled by PyG Planetoid. Text preprocessing deferred to Phase 5."}


def clean_text(text: str) -> str:
    """Clean and normalize raw text for BERT encoding (Phase 5).

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    return text.strip().lower()


def build_graph(edge_list: List[tuple], num_nodes: int):
    """Construct PyG edge_index from a list of (src, dst) tuples.

    Args:
        edge_list: List of (source, destination) node index tuples.
        num_nodes: Total number of nodes.

    Returns:
        edge_index tensor [2, num_edges] (undirected).
    """
    import torch
    from torch_geometric.utils import to_undirected

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return edge_index


def extract_text_features(texts: List[str], method: str = "tfidf"):
    """Extract initial node features from text attributes.

    Args:
        texts: List of raw text strings per node.
        method: Feature extraction method ("tfidf", "bow", "bert").
                BERT will be implemented in Phase 5.
    """
    if method == "bert":
        raise NotImplementedError("BERT feature extraction will be implemented in Phase 5 (text-graph fusion).")

    from sklearn.feature_extraction.text import TfidfVectorizer

    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        return vectorizer.fit_transform(texts).toarray()
    elif method == "bow":
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", use_idf=False)
        return vectorizer.fit_transform(texts).toarray()
    else:
        raise ValueError(f"Unknown method: {method}")


__all__ = ["preprocess_data", "clean_text", "build_graph", "extract_text_features"]
