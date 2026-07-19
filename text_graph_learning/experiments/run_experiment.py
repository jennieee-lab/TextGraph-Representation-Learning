"""
Experiment Runner — Main Entry Point
======================================

Orchestrates the full experiment pipeline:
  1. Load configuration (YAML)
  2. Set random seed
  3. Load and preprocess dataset
  4. Build model
  5. Train model
  6. Evaluate model
  7. Save results

Usage:
    python experiments/run_experiment.py --config configs/default.yaml --model gcn
    python experiments/run_experiment.py --model mlp
    python experiments/run_experiment.py --model graphsage --dataset cora
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, get_device
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.data.dataset import load_dataset, get_dataset_info
from src.models import build_model
from src.trainers import Trainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a graph representation learning experiment on citation networks."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name (mlp, gcn, graphsage).",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Override dataset name (cora, citeseer, pubmed).",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Override results output directory.",
    )
    return parser.parse_args()


def run_single_experiment(config: dict, model_name: str, logger) -> dict:
    """Run one experiment with a given model and return results."""
    # Override config with model name
    config = json.loads(json.dumps(config))  # deep copy
    config["model"]["name"] = model_name

    # Seed
    set_seed(config["experiment"]["seed"])

    # Device
    device = get_device(config["experiment"].get("device", "auto"))
    logger.info(f"Using device: {device}")

    # Data
    data = load_dataset(config)
    info = get_dataset_info(data)
    logger.info(f"Dataset: {config['dataset']['name']} | {info}")

    # Model
    model = build_model(config, num_features=info["num_features"], num_classes=info["num_classes"])
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.name} | Parameters: {num_params:,}")

    # Train
    start = time.time()
    trainer = Trainer(model, config, device=device)
    trainer.fit(data)
    train_time = time.time() - start

    # Evaluate
    test_metrics = trainer.evaluate(data)
    logger.info(f"Test results ({model.name}): {test_metrics}")

    return {
        "model": model.name,
        "dataset": config["dataset"]["name"],
        "num_params": num_params,
        "best_epoch": trainer.best_epoch,
        "best_val_acc": trainer.best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_micro": test_metrics["f1_micro"],
        "train_time_seconds": round(train_time, 2),
        "history": trainer.history,
    }


def main():
    """Main experiment entry point."""
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides
    if args.model:
        config["model"]["name"] = args.model
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.results_dir:
        config["experiment"]["results_dir"] = args.results_dir

    # Setup results directory
    results_dir = Path(config["experiment"].get("results_dir", "outputs"))
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("experiment", log_dir=str(results_dir))
    logger.info("=" * 60)
    logger.info("Graph Representation Learning Experiment")
    logger.info("=" * 60)

    # Determine which models to run
    model_name = config["model"]["name"].lower()
    if model_name == "all":
        models_to_run = ["mlp", "gcn", "graphsage"]
    else:
        models_to_run = [model_name]

    all_results = []
    for m in models_to_run:
        logger.info("-" * 40)
        result = run_single_experiment(config, m, logger)
        all_results.append(result)

    # Save combined results
    output_file = results_dir / f"results_{config['dataset']['name']}.json"
    # Strip history for the summary file; keep it separately
    summary = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_file}")

    # Save training history
    history_file = results_dir / f"history_{config['dataset']['name']}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(
            {r["model"]: r["history"] for r in all_results},
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"Training history saved to {history_file}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    header = f"{'Model':<15} {'Test Acc':<12} {'F1 Macro':<12} {'F1 Micro':<12} {'Epochs':<10}"
    logger.info(header)
    logger.info("-" * 60)
    for r in all_results:
        row = (
            f"{r['model']:<15} {r['test_accuracy']:.4f}      "
            f"{r['test_f1_macro']:.4f}      {r['test_f1_micro']:.4f}      "
            f"{r['best_epoch']:<10}"
        )
        logger.info(row)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
