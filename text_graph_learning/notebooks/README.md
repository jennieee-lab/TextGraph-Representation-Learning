# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis (EDA), visualization, and debugging.

## Planned Notebooks

| Notebook (future) | Description |
|--------------------|-------------|
| `01_data_exploration.ipynb` | Dataset statistics, graph visualization, text attribute analysis |
| `02_graph_analysis.ipynb` | Degree distribution, connected components, community structure |
| `03_feature_analysis.ipynb` | TF-IDF / BERT feature visualization (t-SNE, PCA) |
| `04_results_visualization.ipynb` | Plot experiment results, compare models, ablation charts |

## Usage

```bash
jupyter notebook notebooks/
```

## Notes

- Notebooks are for exploration only — final experiment runs should use `experiments/run_experiment.py`.
- Notebooks are git-ignored by default (`.ipynb_checkpoints/`); commit only cleaned notebooks.
