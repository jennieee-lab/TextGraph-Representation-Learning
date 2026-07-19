"""
Visualization — Plot Training Curves and Results
==================================================

Generate publication-quality plots from experiment results:
  1. Training curves (loss + accuracy over epochs)
  2. Model comparison bar chart
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict, output_path: str, dataset_name: str = "Cora"):
    """Plot training loss and validation accuracy curves for all models.

    Args:
        history: Dict mapping model name -> list of epoch records.
        output_path: Path to save the figure.
        dataset_name: Dataset name for the title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"MLP": "#D85A30", "GCN": "#185FA5", "GraphSAGE": "#0F6E56"}
    labels = {"MLP": "MLP (no graph)", "GCN": "GCN", "GraphSAGE": "GraphSAGE"}

    # Loss curve
    ax = axes[0]
    for model_name, records in history.items():
        epochs = [r["epoch"] for r in records]
        losses = [r["loss"] for r in records]
        label = labels.get(model_name, model_name)
        color = colors.get(model_name, None)
        ax.plot(epochs, losses, label=label, color=color, linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"Training Loss on {dataset_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Validation accuracy curve
    ax = axes[1]
    for model_name, records in history.items():
        epochs = [r["epoch"] for r in records]
        val_accs = [r["val_acc"] for r in records]
        label = labels.get(model_name, model_name)
        color = colors.get(model_name, None)
        ax.plot(epochs, val_accs, label=label, color=color, linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title(f"Validation Accuracy on {dataset_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(results: list, output_path: str, dataset_name: str = "Cora"):
    """Plot bar chart comparing test accuracy across models.

    Args:
        results: List of result dicts (from results JSON).
        output_path: Path to save the figure.
        dataset_name: Dataset name for the title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models = [r["model"] for r in results]
    accuracies = [r["test_accuracy"] * 100 for r in results]
    f1_macros = [r["test_f1_macro"] * 100 for r in results]

    colors = {"MLP": "#D85A30", "GCN": "#185FA5", "GraphSAGE": "#0F6E56"}
    bar_colors = [colors.get(m, "#888888") for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, accuracies, width, label="Test Accuracy",
                   color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, f1_macros, width, label="F1 (Macro)",
                   color=bar_colors, alpha=0.5, edgecolor="white", linewidth=0.5)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f"{height:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(f"Model Comparison on {dataset_name} Node Classification", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all visualizations from saved experiment results."""
    results_dir = Path("outputs")
    dataset_name = "cora"

    # Load results
    results_file = results_dir / f"results_{dataset_name}.json"
    history_file = results_dir / f"history_{dataset_name}.json"

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        sys.exit(1)

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    vis_dir = Path("visualization")
    vis_dir.mkdir(parents=True, exist_ok=True)

    dataset_display = dataset_name.capitalize()
    plot_training_curves(history, str(vis_dir / "training_curves.png"), dataset_display)
    plot_model_comparison(results, str(vis_dir / "model_comparison.png"), dataset_display)

    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()
