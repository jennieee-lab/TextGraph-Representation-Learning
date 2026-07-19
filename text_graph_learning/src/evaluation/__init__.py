"""
Evaluation Module — Metrics and Benchmarking
==============================================

This module provides evaluation metrics for graph learning tasks:
  - Node classification: Accuracy, Macro-F1, Micro-F1
  - Link prediction: AUC-ROC, AP (Average Precision), Hits@K

Future Implementation Plan:
    - compute_metrics():  Unified metric computation
    - Task-specific helpers (node classification, link prediction)
"""

from .metrics import compute_metrics, accuracy, f1_score

__all__ = ["compute_metrics", "accuracy", "f1_score"]
