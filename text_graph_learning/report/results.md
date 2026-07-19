# Experiment Results — Node Classification on Cora

## Dataset: Cora Citation Network

The Cora dataset consists of scientific publications classified into one of seven research fields.
- **Nodes:** 2,708 scientific papers
- **Edges:** 10,556 citation links (undirected, after symmetrization)
- **Features:** 1,433-dimensional bag-of-words per node
- **Classes:** 7 research areas (Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory)
- **Splits:** 140 train / 500 validation / 1,000 test (standard Planetoid split)
- **Task:** Transductive node classification

## Models

| Model | Type | Description |
|-------|------|-------------|
| MLP | Non-graph baseline | 2-layer feedforward network using only node features. No graph structure used. |
| GCN | Spectral GNN | Graph Convolutional Network (Kipf & Welling, ICLR 2017). 1st-order Chebyshev approximation of spectral graph convolution. |
| GraphSAGE | Spatial GNN | Inductive representation learning (Hamilton et al., NeurIPS 2017). Mean aggregation over sampled neighborhood. |

## Results

| Model | Test Accuracy | F1 (Macro) | F1 (Micro) | Best Epoch | Parameters | Train Time |
|-------|:------------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| MLP | 58.40% | 56.58% | 58.40% | 52 | 92,231 | 2.4s |
| GCN | **81.80%** | **80.68%** | **81.80%** | 46 | 92,231 | 3.1s |
| GraphSAGE | 78.00% | 77.66% | 78.00% | 21 | 184,391 | 5.9s |

## Key Observations

1. **Graph structure matters — a lot.** GCN outperforms MLP by **+23.4%** accuracy (81.80% vs 58.40%), confirming that leveraging citation topology dramatically improves node classification. MLP, which uses only bag-of-words features without graph structure, quickly overfits to the 140 training nodes and fails to generalize.

2. **GCN vs GraphSAGE.** Both GNNs aggregate neighborhood information, but differ in mechanism. GCN uses spectral graph convolution (normalizing by the graph Laplacian), while GraphSAGE uses inductive mean aggregation. On Cora's transductive setting, GCN's spectral approach yields +3.8% higher accuracy. GraphSAGE's advantage is inductive learning — generalizing to unseen nodes — which is not tested here.

3. **Convergence behavior.** GraphSAGE converges fastest (best validation at epoch 21), but to a lower accuracy. GCN converges at epoch 46. MLP overfits early (training accuracy hits 100% by epoch 20, but validation plateaus at ~59%).

4. **GCN matches published benchmarks.** Our GCN accuracy of 81.80% is consistent with the original Kipf & Welling (2017) paper's reported ~81.5% on Cora, validating our implementation.

## Visualizations

- **Training curves:** `visualization/training_curves.png` — Loss and validation accuracy over epochs for all three models.
- **Model comparison:** `visualization/model_comparison.png` — Bar chart comparing test accuracy and F1 scores.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 64 |
| Number of layers | 2 |
| Dropout | 0.5 |
| Learning rate | 0.01 |
| Weight decay | 5e-4 |
| Optimizer | Adam |
| Max epochs | 200 |
| Early stopping patience | 20 |
| Random seed | 42 |
| Feature normalization | Row-normalization (PyG NormalizeFeatures) |

## Reproducibility

```bash
# Run all three models on Cora
python experiments/run_experiment.py --config configs/default.yaml

# Run a single model
python experiments/run_experiment.py --model gcn

# Generate visualizations
python visualization/plot_results.py
```

## Future Work

- **Text-Graph Fusion (Phase 5):** Integrate BERT text embeddings with GNN structural embeddings. Cora nodes have raw text attributes (paper titles/abstracts) that are currently compressed into bag-of-words — BERT can extract richer semantic features.
- **GAT (Phase 4):** Add Graph Attention Network (Veličković et al., 2018) with learned attention weights for neighbor aggregation.
- **Graph Self-Supervised Learning:** Explore contrastive learning (GraphCL-style) for pretraining — directly related to Prof. Yuning You's foundational work on graph foundation models (GraphCL, NeurIPS 2020, 4800+ citations).
- **Larger datasets:** Evaluate on ogbn-arxiv and heterogeneous networks with rich text attributes.
