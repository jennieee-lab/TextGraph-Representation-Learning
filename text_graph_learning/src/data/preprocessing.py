"""
Preprocessing Module — Data Preprocessing & Graph Construction
===============================================================

Purpose:
    Transform raw datasets into structured graph representations
    with both topological and textual features.

Pipeline:
    1. Load raw data (text documents, citation links, node metadata)
    2. Clean and tokenize text attributes
    3. Construct graph (nodes, edges, adjacency)
    4. Extract initial node features (TF-IDF, BoW, or placeholder)
    5. Save processed data to data/processed/

Future Implementation Plan:
    - preprocess_data():          Main entry point — orchestrates full pipeline
    - clean_text():               Text normalization (lowercase, strip, remove noise)
    - build_graph():              Construct edge_index from citation/edge lists
    - extract_text_features():    TF-IDF / BoW / BERT feature extraction
    - create_splits():            Generate train/val/test masks

Usage (future):
    from src.data.preprocessing import preprocess_data
    preprocess_data(config)
"""

from typing import Any, Dict, List

# TODO: Import libraries when implementing
# import numpy as np
# import torch
# from torch_geometric.utils import to_undirected


def preprocess_data(config: Dict[str, Any]):
    """
    Run the full preprocessing pipeline.

    Args:
        config: Configuration dictionary from YAML.

    Pipeline:
        raw data → text cleaning → graph construction → feature extraction → save

    TODO:
        - Orchestrate sub-steps
        - Save processed data as .pt files in data/processed/
    """
    raise NotImplementedError("Preprocessing will be implemented in Phase 2.")


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text.

    TODO:
        - Lowercase, strip whitespace
        - Remove special characters / URLs
        - Handle missing or empty text
    """
    raise NotImplementedError


def build_graph(edge_list: List[tuple], num_nodes: int):
    """
    Construct PyG edge_index from a list of (src, dst) tuples.

    TODO:
        - Convert to tensor format
        - Make undirected (for undirected graphs)
        - Remove self-loops if needed
    """
    raise NotImplementedError


def extract_text_features(texts: List[str], method: str = "tfidf"):
    """
    Extract initial node features from text attributes.

    Args:
        texts: List of raw text strings per node.
        method: Feature extraction method ("tfidf", "bow", "bert").

    TODO:
        - Support TF-IDF for baseline
        - Support BERT for text-enhanced models
    """
    raise NotImplementedError
