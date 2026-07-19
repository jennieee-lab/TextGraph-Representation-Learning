"""
Trainers Module — Training Loop for Node Classification
========================================================

Full-batch transductive training loop for GNN models on citation networks.
Supports early stopping, checkpointing, and metric logging.
"""

import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.metrics import compute_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Full-batch trainer for transductive node classification.

    Standard training setup for citation networks (Cora, Citeseer, PubMed):
      - Full-batch gradient descent (entire graph fits in memory)
      - Cross-entropy loss on training nodes
      - Validation-based early stopping
      - Evaluation on test nodes

    Args:
        model: PyTorch model instance.
        config: Configuration dictionary.
        device: Device to train on ("cpu" or "cuda").
    """

    def __init__(self, model, config: Dict[str, Any], device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device

        train_cfg = config["training"]
        self.epochs = train_cfg.get("epochs", 200)
        self.lr = train_cfg.get("learning_rate", 0.01)
        self.weight_decay = train_cfg.get("weight_decay", 5e-4)
        self.patience = train_cfg.get("early_stopping_patience", 20)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.history: List[Dict[str, float]] = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.best_state = None

    def fit(self, data):
        """Train the model with early stopping on validation accuracy.

        Args:
            data: PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask.
        """
        data = data.to(self.device)
        self.model.train()
        patience_counter = 0

        logger.info(
            f"Training {self.model.name} for up to {self.epochs} epochs "
            f"(lr={self.lr}, wd={self.weight_decay}, patience={self.patience})"
        )

        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()

            # Evaluate
            train_metrics = self._eval_node_classification(data, data.train_mask)
            val_metrics = self._eval_node_classification(data, data.val_mask)

            epoch_record = {
                "epoch": epoch,
                "loss": loss.item(),
                "train_acc": train_metrics["accuracy"],
                "val_acc": val_metrics["accuracy"],
            }
            self.history.append(epoch_record)

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_epoch = epoch
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                    f"Train Acc {train_metrics['accuracy']:.4f} | "
                    f"Val Acc {val_metrics['accuracy']:.4f}"
                )

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch} (no val improvement for {self.patience} epochs)")
                break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")

    def evaluate(self, data) -> Dict[str, float]:
        """Evaluate model on the test set.

        Args:
            data: PyG Data object.

        Returns:
            Dictionary with accuracy, f1_macro, f1_micro on test nodes.
        """
        data = data.to(self.device)
        return self._eval_node_classification(data, data.test_mask)

    @torch.no_grad()
    def _eval_node_classification(self, data, mask) -> Dict[str, float]:
        """Compute metrics on a subset of nodes defined by mask."""
        self.model.eval()
        out = self.model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1).cpu().numpy()
        true = data.y[mask].cpu().numpy()
        return compute_metrics(true, pred, task="node_classification")


__all__ = ["Trainer"]
