# Trainers Module

This directory contains training logic and experiment orchestration.

## Planned Components

| Component | File (future) | Description |
|-----------|---------------|-------------|
| **Trainer** | `trainer.py` | Base trainer class — manages training loop, validation, checkpointing, and logging. |
| **GNNTrainer** | `gnn_trainer.py` | Trainer for transductive GNN models (GCN, GAT). Full-batch training over the entire graph. |
| **InductiveTrainer** | `inductive_trainer.py` | Trainer for inductive models (GraphSAGE). Mini-batch neighbor sampling. |
| **FusionTrainer** | `fusion_trainer.py` | Trainer for TextGraphFusion — handles joint text encoder + GNN optimization. |
| **Callback** | `callbacks.py` | Early stopping, LR scheduling, checkpoint callbacks. |

## Training Pipeline

```
config → Trainer(model, config)
              │
              ├── fit(train_data, val_data)
              │     ├── epoch loop
              │     │   ├── forward pass
              │     │   ├── compute loss
              │     │   ├── backward pass
              │     │   └── optimizer step
              │     ├── validate (val_data)
              │     ├── early stopping check
              │     └── log to TensorBoard
              │
              └── evaluate(test_data)
                    └── compute final metrics
```

## Notes

- Trainers should be model-agnostic where possible — the model defines forward(),
  the trainer handles the training mechanics.
- Checkpoint paths are read from config (experiment.checkpoint_dir).
- TensorBoard logs go to experiment.log_dir.
