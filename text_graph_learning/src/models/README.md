# Models Module

This directory contains all model implementations for the research project.

## Model Overview

### Baselines

| Model | File (future) | Description |
|-------|---------------|-------------|
| **MLP** | `mlp.py` | Multi-layer perceptron using node features only. Serves as a structural-agnostic baseline — if GNNs can't beat MLP, the graph structure isn't helping. |
| **Node2Vec** | `node2vec.py` | Random-walk based embedding (Grover & Leskovec, 2016). Learns structural similarity via biased random walks + SkipGram. Captures topology without node features. |

### Graph Neural Networks

| Model | File (future) | Description |
|-------|---------------|-------------|
| **GCN** | `gcn.py` | Graph Convolutional Network (Kipf & Welling, 2017). Spectral approach using first-order Chebyshev approximation. Aggregates neighbors with symmetric normalized mean. |
| **GraphSAGE** | `graphsage.py` | Inductive representation learning (Hamilton et al., 2017). Samples a fixed-size neighborhood and aggregates (mean/LSTM/pool). Supports inductive inference on unseen nodes. |
| **GAT** | `gat.py` | Graph Attention Network (Velickovic et al., 2018). Learns attention weights per neighbor, enabling adaptive importance. Multi-head attention improves stability. |

### Text Encoder

| Model | File (future) | Description |
|-------|---------------|-------------|
| **BERT Encoder** | `text_encoder.py` | Pretrained BERT (Devlin et al., 2019) fine-tuned on node text attributes. Produces 768-dim semantic embeddings. Can be frozen or fine-tuned end-to-end. |

### Fusion Model

| Model | File (future) | Description |
|-------|---------------|-------------|
| **TextGraphFusion** | `text_graph_fusion.py` | Combines BERT text embeddings with GNN structural embeddings. Fusion strategies: (1) Concatenation + MLP, (2) Gated fusion, (3) Cross-attention. This is the core contribution of the project. |

## Architecture Diagram (Conceptual)

```
                         TextGraphFusion
                        ┌──────────────────┐
                        │  Fusion Layer     │
                        │  (concat / gate   │
                        │   / cross-attn)   │
                        └──────┬─────┬─────┘
                               │     │
                 ┌─────────────┘     └─────────────┐
                 ▼                                  ▼
        ┌─────────────────┐              ┌─────────────────┐
        │  GNN Encoder    │              │  Text Encoder   │
        │  (GCN/SAGE/GAT) │              │  (BERT)         │
        └────────┬────────┘              └────────┬────────┘
                 │                                │
          Graph topology                        Raw text
          (edge_index)                     (node attributes)
```

## Implementation Notes

- All models will inherit from a common `BaseModel` class (to be created).
- Each model file will contain a model class + a factory function.
- Models will read hyperparameters from the config, not from hardcoded values.
