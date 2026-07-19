"""
Metrics Module — Evaluation Metrics for Graph Learning Tasks
==============================================================

Standard evaluation metrics for node classification:
  - Accuracy
  - Macro-F1, Micro-F1
"""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score as sklearn_f1


def accuracy(y_true, y_pred) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def f1_score(y_true, y_pred, average: str = "macro") -> float:
    """Compute F1 score (macro or micro).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: "macro" or "micro".

    Returns:
        F1 score in [0, 1].
    """
    return float(sklearn_f1(y_true, y_pred, average=average))


def compute_metrics(y_true, y_pred, task: str = "node_classification", **kwargs) -> Dict[str, float]:
    """Compute evaluation metrics for a given task.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        task: One of "node_classification" or "link_prediction".

    Returns:
        Dictionary of metric name -> value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if task == "node_classification":
        return {
            "accuracy": accuracy(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_micro": f1_score(y_true, y_pred, average="micro"),
        }
    else:
        raise ValueError(f"Unknown task: {task}")


__all__ = ["compute_metrics", "accuracy", "f1_score"]
