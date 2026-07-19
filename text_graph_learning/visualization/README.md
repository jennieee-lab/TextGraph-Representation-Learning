# Visualization

This directory contains scripts and outputs for generating publication-quality figures and plots.

## Planned Visualizations

| Script (future) | Description |
|------------------|-------------|
| `plot_training_curves.py` | Training/validation loss and accuracy curves |
| `plot_embeddings.py` | t-SNE / UMAP visualization of learned node embeddings |
| `plot_comparison.py` | Bar charts comparing model performance across tasks |
| `plot_ablation.py` | Ablation study results (fusion strategy comparison) |
| `plot_graph_structure.py` | Network topology visualization (for small graphs) |

## Tools

- **Matplotlib**: Primary plotting library
- **TensorBoard**: Real-time training monitoring
- **NetworkX**: Graph structure visualization

## Output Format

- All figures saved as both `.png` (for reports) and `.pdf` (for LaTeX papers).
- Default DPI: 300 for publication quality.
- Color scheme: Consistent across all figures for a unified visual identity.
