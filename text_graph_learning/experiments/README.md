# Experiments

This directory contains experiment scripts that orchestrate the full research pipeline.

## Structure

| File | Description |
|------|-------------|
| `run_experiment.py` | Main entry point — loads config, builds model, trains, evaluates, saves results. |

## Planned Experiments

| Script (future) | Experiment | Models | Task |
|-----------------|------------|--------|------|
| `run_experiment.py` | General experiment runner | Any (via config) | Any (via config) |
| `baseline_comparison.py` | Exp-1: Baseline vs. GNN comparison | MLP, Node2Vec, GCN, GraphSAGE, GAT | Node classification |
| `fusion_evaluation.py` | Exp-2: Text-graph fusion evaluation | TextGraphFusion vs. GNN-only | Node classification |
| `link_prediction.py` | Exp-3: Link prediction | All models | Link prediction |
| `ablation_study.py` | Exp-4: Fusion strategy ablation | Fusion variants (early/late/cross-attn) | Node classification |
| `generalization_test.py` | Exp-5: Inductive generalization | Best model | Unseen nodes |

## Usage (Future)

```bash
# Run a single experiment
python experiments/run_experiment.py --config configs/default.yaml

# Run baseline comparison
python experiments/baseline_comparison.py --config configs/default.yaml
```

## Output

Experiment results (metrics, logs, checkpoints) are saved to:
- `outputs/` — Result files (JSON/CSV)
- `runs/` — TensorBoard logs
- `checkpoints/` — Model weights

All output directories are git-ignored.
