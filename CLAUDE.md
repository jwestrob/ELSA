# CLAUDE.md

⚠️  **IMPORTANT**: Only let the human run bash commands. Do not run long-running commands like `elsa embed` or `elsa build` - they will timeout.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ELSA (Embedding Locus Shingle Alignment) is a bioinformatics tool for order-aware syntenic-block discovery from protein language-model embeddings. It processes assembled genomes or metagenomes to find similar syntenic blocks via collinearity chaining.

**Key Architecture:**
- **GPU-optimized** for M4 Max MacBook Pro (48GB unified memory) with CPU fallback
- Memory-mapped arrays for large datasets 
- Dual indexing system: discrete MinHash + continuous signed-random-projection
- Pipeline stages: Ingest → PLM embeddings → Projection → Shingling → Indexing → Search

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