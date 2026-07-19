# Text-enhanced Graph Representation Learning for Knowledge Discovery in Heterogeneous Networks

## 1. Project Overview

This project investigates how **graph structural information** and **textual semantic information** can be jointly learned to produce richer node representations for downstream graph learning tasks such as node classification and link prediction.

Real-world data — citation networks, social networks, knowledge graphs — naturally form heterogeneous graphs where entities carry rich textual attributes (abstracts, profiles, descriptions). Traditional graph neural networks (GNNs) leverage topology but often treat node features as static embeddings, underutilizing the semantic signal in raw text. Conversely, text encoders capture semantics but are blind to relational structure.

This project explores **text-graph fusion models** that bridge this gap, aiming to advance the frontier of Graph Representation Learning, Foundation Models, and Knowledge Discovery.

---

## 2. Research Question

> **How can graph structures and textual information be effectively combined to learn better representations for downstream graph learning tasks?**

Sub-questions:
- What is the contribution of textual semantics versus structural topology across different graph types?
- Which fusion strategies (early, late, cross-attention) yield the best performance-cost trade-off?
- Can text-enhanced representations generalize to unseen nodes and unseen graph structures?

---

## 3. Motivation

- **Graph-only methods miss semantics.** GNNs propagate messages along edges but often rely on shallow or hand-crafted features, missing the rich semantic content in node text attributes.
- **Text-only methods miss structure.** Language models encode node descriptions but cannot capture multi-hop relational dependencies that define graph topology.
- **Joint learning is underexplored.** While recent works (e.g., G-Retriever, GraphFormers) explore text-graph integration, the optimal fusion strategy remains an open question — especially for heterogeneous networks.
- **Practical impact.** Better node representations directly improve knowledge discovery tasks: entity resolution, relation extraction, recommendation, and scientific literature navigation.

---

## 4. Planned Methodology

### 4.1 Baselines
| Model | Type | Description |
|-------|------|-------------|
| MLP | Non-graph | Feedforward network using node features only; establishes a structural-agnostic lower bound. |
| Node2Vec | Structural | Random-walk based embedding; captures structural similarity without node features. |

### 4.2 Graph Neural Networks
| Model | Key Mechanism |
|-------|---------------|
| GCN | Spectral graph convolution (Chebyshev approximation) |
| GraphSAGE | Inductive neighbor sampling and aggregation |
| GAT | Attention-weighted neighbor aggregation |

### 4.3 Text Encoder
| Model | Role |
|-------|------|
| BERT | Encodes raw node text attributes into dense semantic embeddings |

### 4.4 Fusion Architecture
| Model | Fusion Strategy |
|-------|-----------------|
| TextGraphFusion | Combines BERT text embeddings with GNN structural embeddings via concatenation, projection, or cross-attention |

### 4.5 Pipeline
```
Raw Data → Preprocessing → Graph Construction → Text Feature Extraction
    → GNN Encoding → Text-Graph Fusion → Task-Specific Head → Evaluation
```

---

## 5. Future Experiments

| Experiment | Task | Models | Dataset |
|------------|------|--------|---------|
| Exp-1 | Node classification | MLP, Node2Vec, GCN, GraphSAGE, GAT | Citation network (e.g., Cora, PubMed) |
| Exp-2 | Node classification | TextGraphFusion vs. GNN-only | Heterogeneous network with text attributes |
| Exp-3 | Link prediction | All models | Same as above |
| Exp-4 (ablation) | Node classification | Fusion variants (early/late/cross-attention) | Same as above |
| Exp-5 | Generalization | Best model | Inductive setting (unseen nodes) |

---

## 6. Technology Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch, PyTorch Geometric (PyG) |
| Text Encoding | HuggingFace Transformers (BERT) |
| Scientific Computing | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, TensorBoard |
| Configuration | YAML |
| Environment | Conda / pip |

---

## 7. Project Roadmap

```
Phase 1 — Project Foundation          [complete]
  └─ Skeleton, configs, documentation, environment setup

Phase 2 — Data Pipeline               [complete]
  └─ Cora dataset loading via PyG Planetoid, feature normalization

Phase 3 — Baseline Models             [complete]
  └─ MLP (non-graph baseline) implemented & evaluated

Phase 4 — GNN Models                  [complete]
  └─ GCN, GraphSAGE implemented & evaluated on Cora

Phase 5 — Text-Graph Fusion           [planned]
  └─ BERT encoder + fusion architecture

Phase 6 — Experiments & Ablation      [partial]
  └─ Baseline comparison done; ablation & generalization pending

Phase 7 — Analysis & Report           [partial]
  └─ Results documented; full report pending
```

---

## 8. Results

### Node Classification on Cora

| Model | Test Accuracy | F1 (Macro) | F1 (Micro) | Parameters |
|-------|:------------:|:----------:|:----------:|:----------:|
| MLP (no graph) | 58.40% | 56.58% | 58.40% | 92K |
| GCN | **81.80%** | **80.68%** | **81.80%** | 92K |
| GraphSAGE | 78.00% | 77.66% | 78.00% | 184K |

**Key findings:**
- GCN outperforms MLP by **+23.4%**, confirming that graph topology is critical for node classification.
- GCN's 81.80% accuracy matches the original Kipf & Welling (ICLR 2017) benchmark, validating the implementation.
- GraphSAGE converges faster (best at epoch 21) but to lower accuracy than GCN on this transductive task.

**Visualizations:**
- [Training curves](visualization/training_curves.png) — Loss and validation accuracy over epochs
- [Model comparison](visualization/model_comparison.png) — Bar chart of test accuracy and F1

Full results: [report/results.md](report/results.md)

---

## 9. Next Steps & Research Directions

This project establishes supervised graph representation learning foundations. The following directions are planned for future exploration:

### Text-Graph Fusion (Phase 5)
- Integrate a pretrained BERT encoder to produce semantic node embeddings from raw text attributes (e.g., paper titles/abstracts in Cora)
- Compare fusion strategies: early concatenation, late fusion, cross-attention
- Evaluate whether textual semantics improve performance on graphs with rich text attributes

### Graph Self-Supervised Learning
- Extend the current supervised pipeline toward **self-supervised pre-training** on graphs, building on methods such as GraphCL (You et al., NeurIPS 2020) and Deep Graph Infomax (DGI, Veličković et al., ICLR 2019)
- Explore graph augmentations (node dropping, edge perturbation, attribute masking) and contrastive objectives to learn representations without labels
- Investigate transferability of pretrained graph encoders to downstream node/graph-level tasks

### Inductive Generalization
- Evaluate model performance on unseen nodes and unseen graphs (inductive setting), where GraphSAGE's design should outperform transductive GCN

### Broader Impact
- Explore applications of graph representation learning to **biological networks** (e.g., protein-protein interaction networks, cell-cell communication graphs), where graph structure encodes meaningful relational priors — connecting graph foundation models to AI for life sciences

---

## Repository Structure

```
text_graph_learning/
├── README.md                 ← This file
├── requirements.txt          ← Python dependencies
├── environment.yml           ← Conda environment
├── .gitignore
├── configs/
│   └── default.yaml          ← Experiment configuration
├── data/
│   ├── raw/                  ← Original datasets (git-ignored)
│   ├── processed/            ← Preprocessed data (git-ignored)
│   └── README.md
├── src/
│   ├── data/                 ← Dataset loading & preprocessing
│   ├── models/               ← Model definitions
│   ├── trainers/             ← Training loops
│   ├── evaluation/           ← Metrics & evaluation
│   └── utils/                ← Utilities (logging, seeding)
├── experiments/
│   └── run_experiment.py     ← Main experiment entry point
├── notebooks/                ← Exploratory analysis
├── visualization/            ← Plot generation
└── report/
    └── research_report_outline.md
```

---

## Getting Started

### Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate text_graph_learning

# Or use pip
pip install -r requirements.txt
```

### Run Experiments

```bash
# Run all three models (MLP, GCN, GraphSAGE) on Cora
python experiments/run_experiment.py --config configs/default.yaml

# Run a single model
python experiments/run_experiment.py --model gcn

# Generate visualizations (training curves + model comparison)
python visualization/plot_results.py
```

The Cora dataset is automatically downloaded on first run via PyG's Planetoid loader. If the download fails (network issues), place the raw files in `data/raw/Cora/raw/` — see [PyG Planetoid docs](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html) for details.

---

## License

This project is developed for academic research purposes.

## Contact

For questions related to this research project, please refer to the repository issues.
