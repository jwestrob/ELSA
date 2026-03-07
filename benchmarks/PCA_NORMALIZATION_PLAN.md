# PCA & Normalization Technique Comparison

## Goal

Find the optimal embedding projection for ELSA's synteny detection pipeline.
All techniques use the same UniRef50 subsample (52,949 proteins) to fit
transforms, then evaluate on the established E. coli operon benchmark (20
genomes, 58 operons, 10,182 instances).

## Techniques

### 1. Frozen PCA (new default)
- Standard PCA fitted on UniRef50 subsample, 480D -> 256D
- L2 normalize after projection
- This is what we're currently computing
- **Rationale**: Generalizable projection space, not overfit to any single dataset

### 2. Frozen PCA + Whitening
- Same PCA but with `whiten=True` (divide by sqrt of eigenvalues)
- Decorrelates dimensions AND equalizes variance across components
- L2 normalize after
- **Rationale**: Prevents top PCs from dominating cosine similarity; may improve
  retrieval by spreading information more evenly across dimensions

### 3. All-but-the-top PCA
- Remove the top-k principal components (k=1 or k=3) before projection
- These top components often encode token frequency / sequence length bias
  rather than functional information (Mu et al., 2018)
- Steps: center, project out top-k PCs, then PCA to 256D on residual
- L2 normalize after
- **Rationale**: PLM embeddings have an anisotropic distribution dominated by
  a few directions. Removing them may improve cosine similarity quality.

### 4. OPQ (FAISS Optimized Product Quantization)
- FAISS-native rotation optimized for nearest-neighbor retrieval
- Train OPQ rotation matrix on UniRef50 embeddings
- Can combine with dimensionality reduction (OPQ480,256 or OPQ256,256)
- No separate L2 norm needed — OPQ optimizes for inner product / L2 search
- **Rationale**: Directly optimizes the projection for the retrieval task we
  actually use (HNSW neighbor search), rather than variance preservation

### 5. Whitened raw embeddings (480D baseline)
- No dimensionality reduction — keep full 480D
- Center + whiten (StandardScaler or PCA whiten without reduction)
- L2 normalize after
- **Rationale**: Baseline to measure whether dimensionality reduction helps
  or hurts. 480D is still tractable for HNSW.

## Evaluation Protocol

For each technique:

1. **Fit transform** on UniRef50 raw embeddings (`data/frozen_pca/uniref50_raw.parquet`)
2. **Apply transform** to E. coli benchmark raw embeddings (need `--save-raw` from existing run, or re-embed)
3. **Run chain pipeline** (`elsa analyze`) with default parameters
4. **Evaluate**:
   - Operon recall (strict, independent, any coverage) against RegulonDB
   - Ortholog validation (OrthoFinder orthogroup overlap)
   - Block count, mean block size, cluster count
   - HNSW neighbor quality (mean cosine of top-50 neighbors)

### Results (March 2026)

All techniques evaluated on 85,809 E. coli proteins (20 genomes), 10,182 operon
instances from RegulonDB. Same gene set for all techniques — fair comparison.

| Technique | Dim | Strict | Independent | Any Cov | Blocks | Clusters | Mean Size |
|-----------|-----|--------|-------------|---------|--------|----------|-----------|
| Per-dataset PCA | 256 | 12.4% | 76.1% | 98.1% | 25,730 | 7,734 | 22.1 |
| **Frozen PCA** | 256 | 12.4% | **76.1%** | 98.0% | 25,692 | 7,727 | 22.1 |
| Frozen PCA + whiten | 256 | 12.4% | 75.9% | 98.0% | 25,632 | 7,698 | 22.2 |
| ABT k=1 | 256 | 12.4% | **76.1%** | 98.0% | 25,688 | 7,712 | 22.2 |
| ABT k=3 | 256 | 12.4% | 75.8% | 98.0% | 25,685 | 7,675 | 22.1 |
| **OPQ** | 256 | 12.4% | **76.1%** | **98.1%** | 25,686 | 7,726 | 22.2 |
| Whitened raw 480D | 480 | 12.4% | 76.0% | 98.0% | 25,689 | 7,705 | 22.2 |

**Conclusion:** All techniques perform within ±0.3% of each other. The chaining
algorithm dominates performance, not the projection. **Frozen PCA is the right
default** — it matches per-dataset PCA while being generalizable across datasets,
and 256D saves compute/memory vs 480D with no quality loss.

Note: These numbers (76.1% independent) are lower than the 82.6% reported in
CLAUDE.md because this run uses a different gene set (elsa_index_nopca, 85,809
genes) vs the original benchmark (elsa_index, 90,566 genes from a different
gene calling run).

## Prerequisites

- [x] UniRef50 sample embedded (running — `data/frozen_pca/uniref50_raw.parquet`)
- [x] E. coli raw 480D embeddings available: `benchmarks/elsa_output/ecoli/elsa_index_nopca/ingest/genes.parquet` (85,809 proteins, 480 `emb_` cols — these are unprojected ESM2-t12 embeddings, just named `emb_` instead of `raw_`)
- [x] Script to fit all transforms from the same raw embeddings (`scripts/fit_all_projections.py`)
- [ ] Script to apply each transform to E. coli embeddings and produce per-technique `genes.parquet`

## Execution Plan

```bash
# Step 1: Wait for UniRef50 embedding to finish (~overnight Mar 4)
# Monitor: cat data/frozen_pca/embed_progress.txt

# Step 2: Fit all 6 transforms (~1 min)
python scripts/fit_all_projections.py \
    --raw data/frozen_pca/uniref50_raw.parquet \
    -o data/frozen_pca/ --dim 256

# Step 3: Apply each transform to E. coli 480D embeddings
# Source: benchmarks/elsa_output/ecoli/elsa_index_nopca/ingest/genes.parquet
# Note: columns are emb_000..emb_479 (not raw_), same data though
# Output: benchmarks/results/projection_ablation/{technique}/genes.parquet

# Step 4: Run chain pipeline for each technique
# elsa analyze with each projected genes.parquet

# Step 5: Evaluate operon recall for each technique
# Compare against RegulonDB 58 operons
```

## File Layout

```
data/frozen_pca/
├── uniref50_raw.parquet          # Raw 480D embeddings (fitting data)
├── pca_model.pkl                 # Technique 1: standard PCA
├── pca_whiten_model.pkl          # Technique 2: PCA + whitening
├── abt_pca_model.pkl             # Technique 3: all-but-top + PCA
├── abt_components.pkl            # Top-k components to remove
├── opq_transform.pkl             # Technique 4: OPQ rotation
└── whiten_scaler.pkl             # Technique 5: whitening only

benchmarks/results/projection_ablation/
├── frozen_pca/                   # Chain results per technique
├── frozen_pca_whiten/
├── abt_k1/
├── abt_k3/
├── opq/
└── whitened_raw_480/
```

## Implementation Notes

- All transforms are fitted on UniRef50, frozen, then applied to benchmark data
- The `frozen_pca_path` config option already supports loading a pre-fitted model
- Need to generalize to support arbitrary sklearn-compatible transforms
- OPQ requires FAISS — check if installed in elsa env
- For all-but-top: implement as a two-stage transform (remove top-k, then PCA)
