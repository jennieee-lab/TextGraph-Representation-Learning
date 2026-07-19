# Data Directory

This directory stores all datasets used in the project.

## Structure

```
data/
├── raw/           ← Original, unmodified datasets (git-ignored)
├── processed/     ← Preprocessed data ready for model input (git-ignored)
└── README.md      ← This file
```

## Notes

- **Raw data** is never modified. All transformations produce files in `processed/`.
- Both `raw/` and `processed/` are git-ignored to keep the repository lightweight.
- `.gitkeep` files ensure the directory structure is tracked in version control.

## Planned Datasets

| Dataset | Type | Nodes | Edges | Text Attributes | Task |
|---------|------|-------|-------|-----------------|------|
| Cora | Citation network | ~2,700 | ~5,400 | Paper text | Node classification |
| PubMed | Citation network | ~19,700 | ~44,300 | Paper abstracts | Node classification |
| ogbn-arxiv | Citation network | ~169,000 | ~1.2M | Paper titles+abstracts | Node classification |

## Data Pipeline (Future)

```
raw/ → preprocessing.py → processed/
         ↓
    dataset.py → PyG Data objects
```

See `src/data/preprocessing.py` and `src/data/dataset.py` for implementation details.
