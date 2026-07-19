"""
Experiment Runner — Main Entry Point
======================================

Purpose:
    Orchestrate the full experiment pipeline:
      1. Load configuration (YAML)
      2. Set random seed
      3. Load and preprocess dataset
      4. Build model
      5. Train model
      6. Evaluate model
      7. Save results

Usage (future):
    python experiments/run_experiment.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a graph representation learning experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: str):
    """
    Load configuration from a YAML file.

    TODO:
        - Use yaml.safe_load or OmegaConf
        - Support config inheritance / overrides
    """
    raise NotImplementedError("Config loading will be implemented in Phase 2.")


def main():
    """
    Main experiment entry point.

    Pipeline:
        config → seed → data → model → train → evaluate → save

    TODO:
        - Load config from --config path
        - Set random seed (src.utils.seed.set_seed)
        - Load dataset (src.data.dataset.load_dataset)
        - Build model (src.models.build_model)
        - Initialize trainer (src.trainers.Trainer)
        - Train and evaluate
        - Save results to experiment.results_dir
    """
    args = parse_args()
    print(f"[Experiment] Config: {args.config}")
    print("[Experiment] Phase 1 skeleton — full pipeline will be implemented in future phases.")
    print("[Experiment] Nothing to run yet.")


if __name__ == "__main__":
    main()
