# Configuration Guide

This document summarises the keys recognised by the operon embedding CLI. Any
subcommand can load a YAML config via `--config`; values fall back to the
defaults in `config.yaml`.

## Sections

### `preprocess`
- `pca_dims`: target dimensionality after PCA.
- `eps`: eigenvalue stabiliser for whitening.

### `shingle`
- `k`: number of genes per shingle.
- `stride`: slide step.
- `use_positional_moments`: reserved for future positional features.

### `index.hnsw`
- `M`, `ef_construction`, `ef_search`: HNSW construction/query parameters.
- `top_k`: number of neighbours retrieved per shingle (used by `retrieve`).

### `sinkhorn`
- `epsilon`, `iters`, `topK_per_gene`: entropic OT parameters.

### `graph`
- `lambda_cosine`: cosine vs Sinkhorn blend ratio.
- `tau`: temperature applied to Sinkhorn similarities.
- `prune_threshold`: minimum edge weight retained.
- `knn`: reciprocal neighbourhood size.

### `cluster`
- `method`: `leiden` (default) or `hdbscan`.
- `leiden_resolution`: resolution parameter for Leiden.
- `min_cluster_size`, `min_samples`: HDBSCAN settings (used when method is
  `hdbscan`).

### `metric`
Reserved for optional metric-learning integration (milestone 5).

### `evaluate`
- `pair_labels_csv`: optional CSV with ground-truth pair labels.
- `cluster_labels_csv`: optional CSV with ground-truth cluster labels.

### `paths`
Convenience shortcuts so CLI commands can omit explicit arguments:
- `embeddings`, `preprocessor`, `shingles`: gene embeddings and processed
  artefacts.
- `index_dir` (or `hnsw_index`): directory (or file) containing the HNSW index.
- `pair_predictions`: JSON output from `rerank`.
- `cluster_assignments`: JSON output from `cluster`.
- `eval_report`: destination JSON file for the evaluation summary.
- `data_root`, `contig_sizes`, `small_split`: dataset-specific helpers used by
  tests and tooling.

Environment variables referenced in the YAML (e.g. `${OPERON_TEST_DATA}`) are
expanded automatically when loading the config.
