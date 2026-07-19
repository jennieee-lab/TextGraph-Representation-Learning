"""
Metrics Module — Evaluation Metrics for Graph Learning Tasks
==============================================================

Purpose:
    Provide standard evaluation metrics for:
      1. Node classification — Accuracy, Macro-F1, Micro-F1
      2. Link prediction    — AUC-ROC, Average Precision, Hits@K

Future Implementation Plan:
    - compute_metrics():    Unified entry point — dispatches by task type
    - accuracy():           Classification accuracy
    - f1_score():           Macro/Micro F1
    - roc_auc():             AUC-ROC for link prediction
    - average_precision():   AP for link prediction

Usage (future):
    from src.evaluation.metrics import compute_metrics
    metrics = compute_metrics(y_true, y_pred, task="node_classification")
"""

from typing import Any, Dict, Optional

# TODO: Import when implementing
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def compute_metrics(
    y_true,
    y_pred,
    task: str = "node_classification",
    **kwargs,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for a given task.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels (or probabilities for link prediction).
        task: One of "node_classification" or "link_prediction".

    Returns:
        Dictionary of metric name → value.

    TODO:
        - Dispatch to task-specific metric functions
        - Support both transductive and inductive settings
    """
    raise NotImplementedError("Metrics will be implemented in Phase 6.")


def accuracy(y_true, y_pred) -> float:
    """
    Compute classification accuracy.

    TODO: Use sklearn.metrics.accuracy_score
    """
    raise NotImplementedError


def f1_score(y_true, y_pred, average: str = "macro") -> float:
    """
    Compute F1 score (macro or micro).

    TODO: Use sklearn.metrics.f1_score
    """
    raise NotImplementedError
