# CLAUDE.md

⚠️  **IMPORTANT**: Only let the human run bash commands. Do not run long-running commands like `elsa embed` or `elsa build` - they will timeout.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ELSA (Embedding Locus Shingle Alignment) is a bioinformatics tool for order-aware syntenic-block discovery from protein language-model embeddings. It processes assembled genomes or metagenomes to find similar syntenic blocks via collinearity chaining and clusters them via mutual-Jaccard over order-aware shingles.

**Key Architecture:**
- **GPU-optimized** for M4 Max MacBook Pro (48GB unified memory) with CPU fallback
- Memory-mapped arrays for large datasets 
- SRP-based order-aware shingling (strand-insensitive tokens) + mutual-Jaccard clustering
- Minimal default: no hybrid augmentation; optional narrow post-cluster attach (PFAM-agnostic)

## Development Commands

This is a new project - no build/test commands exist yet. Based on the specification, the following CLI commands will need to be implemented:

```bash
# When implemented:
elsa init      # Create starter config and samples template
elsa embed     # Process FASTA to PLM embeddings  
elsa build     # Build discrete and continuous indexes
elsa find      # Find syntenic blocks for query locus
elsa explain   # Show top shingles and aligned window pairs
elsa stats     # QC summaries and collision analysis
```

## Code Structure (To Be Implemented)

The specification outlines this structure:

```
elsa/
├── cli.py              # Main CLI interface
├── api/service.py      # FastAPI service endpoints
├── params.py           # Configuration validation
├── manifest.py         # Data registry management
├── parse_gff.py        # GFF parsing and gene calling
├── embeddings.py       # PLM model interfaces (ProtT5, ESM2)
└── [additional modules per workplan]
```

## Configuration System

- **Main config:** `elsa.config.yaml` - comprehensive parameters for all pipeline stages
- **Data manifest:** `samples.tsv` - tab-delimited sample inventory
- **Work directory:** `./elsa_index/` - default location for all generated artifacts

Key config sections: data, ingest, plm, shingles, discrete, continuous, chain, dtw, score, system

## Data Pipeline Architecture

**Stage sequence:**
1. **Ingest** - Parse FASTA, call genes (Prodigal/MetaProdigal), translate to AA
2. **PLM embeddings** - Load ProtT5/ESM2, batch process proteins, pool embeddings
3. **Projection** - PCA to target dimension D (default 256), L2 normalize
4. **Shingling** - Sliding windows with positional encoding and strand awareness
5. **Discrete indexing** - KMeans codebook, n-gram hashing, MinHash LSH
6. **Continuous indexing** - Signed random projection (SRP) signatures
7. **Registry** - Manifest with hashes, QC plots, resumption checkpoints

**Key file formats:**
- `genes.parquet` - projected protein embeddings
- `windows.parquet` - shingle window embeddings  
- `shingles.parquet` - discrete n-gram hashes
- `srp.parquet` - continuous signatures
- `blocks.jsonl` - search results

## Search Algorithm

1. **Recall** - Query discrete LSH (MinHash) + continuous SRP buckets
2. **Anchor scoring** - Cosine similarity + Jaccard + IDF weighting
3. **Chaining** - Weighted LIS with affine gaps in offset band
4. **Refinement** - Optional DTW banded alignment
5. **Scoring** - Combined anchor strength + chain length - gap penalties

## Clustering (Updated)

We use a multi-stage, cassette-friendly clustering pipeline designed to keep 2–3 gene cassettes while avoiding “fold-only” co-clustering of unrelated loci:

1. Robustness gate (per block)
   - Compute window index sequences and assess collinearity (MAD of diagonal v = q_idx - t_idx).
   - Require: min_anchors (≥4), min_span_genes (≥8), v_mad_max_genes (default 0.5).
   - Cassette mode: allow very small blocks (n ≤ 4) to pass if perfectly collinear (v_mad=0).

2. Order-aware shingling over SRP tokens
   - Tokenize window embeddings (SRP), build k-gram shingles (k=3 by default), normalize orientation.
   - Drop high-DF shingles aggressively (df_max default 30; optional max_df_percentile ban).

3. Similarity and mutual-k graph
   - Compute IDF-weighted Jaccard over shingle sets (idf = log(1 + N/df)).
   - Edge candidates must satisfy:
     - mutual-k top-k condition,
     - Jaccard ≥ jaccard_tau (default 0.75),
     - at least min_low_df_anchors low-DF shingles in intersection (default 3),
     - intersection mean IDF ≥ idf_mean_min (default 1.0).

4. Graph refinement (anti-hub + cohesion filters)
   - Degree cap: keep top-N weighted edges per node (degree_cap default 10).
   - k-core pruning: require node degree ≥ k (k_core_min_degree default 3).
   - Triangle support: keep edges that participate in ≥ T triangles (triangle_support_min default 1).

5. Community detection
   - Build a weighted graph (edge weight = IDF-weighted Jaccard) and detect communities using greedy modularity (NetworkX).
   - Fallback to connected components if community detection is unavailable.

Key config knobs (analyze.clustering.*):
- df_max (int, default 30), max_df_percentile (float in 0–1 or None)
- jaccard_tau (float, default 0.75), mutual_k (int, default 3)
- min_low_df_anchors (int, default 3), idf_mean_min (float, default 1.0)
- v_mad_max_genes (float, default 0.5), enable_cassette_mode (bool, default True), cassette_max_len (int, default 4)
- degree_cap (int, default 10), k_core_min_degree (int, default 3), triangle_support_min (int, default 1)
- use_community_detection (bool, default True), community_method ('greedy')

Notes:
- This approach preserves sensitivity to small cassettes, but heavily penalizes high-frequency “service” signals (e.g., generic HTH regulators, ABC transport machinery, mobile element motifs) through IDF weighting and low-DF anchor requirements.
- For very large components, increase jaccard_tau, min_low_df_anchors, k_core_min_degree, or set max_df_percentile to ban the top-most frequent shingles entirely.

## Technical Constraints

- **Memory**: Memory-mapped arrays, float16 storage, batchable processing within 48GB envelope
- **GPU**: M4 Max MPS acceleration for PLM inference with automatic CPU fallback
- **Determinism**: All randomness seeded from `system.rng_seed`  
- **Resumption**: Idempotent stages with checkpoints
- **Device support**: MPS (Metal Performance Shaders) primary, CUDA/CPU fallback

## Dependencies (To Be Added)

Based on specification requirements:
- **ML/Embedding**: transformers, torch, sklearn
- **Data**: pandas, pyarrow, numpy  
- **Bio**: biopython, prodigal
- **Service**: fastapi, uvicorn
- **CLI**: click, rich (for progress bars)
- **Indexing**: faiss (optional HNSW), datasketch (MinHash)

## Testing Strategy

- **Unit tests**: GFF parsing, PCA shapes, SRP stability, MinHash determinism
- **Property tests**: Chaining invariance, DTW constraints, Jaccard unbiasedness  
- **Metamorphic**: Gene boundary jitter tolerance
- **Golden set**: Hand-curated loci with expected block ranges
