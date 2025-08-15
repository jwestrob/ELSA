# ELSA: Embedding Locus Shingle Alignment

Order-aware syntenic-block discovery from protein language-model embeddings.

## Quick Start

### 1. Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate elsa

# Install ELSA
pip install -e .
```

### 2. Initialize workspace

```bash
elsa init
```

This creates `elsa.config.yaml` with sensible defaults.

### 3. Run embedding pipeline

```bash
# Basic usage - process all *.fasta files in data/ directory
elsa embed data/

# Supports multiple FASTA extensions automatically: .fasta, .fa, .fna
# Or specify custom pattern:
elsa embed data/ --fasta-pattern "*.genomic.fna"

# ELSA automatically:
# 1. Calls genes with Prodigal on each nucleotide FASTA
# 2. Creates .gff and .faa files alongside the input
# 3. Generates protein embeddings and builds indexes
```

### 4. Build indexes

```bash
elsa build
```

### 5. Find syntenic blocks

```bash
elsa find "strain001_contig1_1:100000"
```

## Pipeline Architecture

### Stage 1: Protein Ingestion
- **Gene calling**: Prodigal (single/meta mode) for *de novo* gene prediction
- **Automatic output**: Creates .gff and .faa files alongside input nucleotide FASTA
- **Resumption support**: Reuses existing .gff/.faa files if present

### Stage 2: Protein Language Model Embedding
- **ESM2**: Meta's evolutionary scale models (650M or 3B parameters)
- **ProtT5**: Rostlab's protein transformer (XL model)
- **GPU optimization**: MPS acceleration on Apple Silicon, CUDA fallback
- **Memory management**: Adaptive batch sizing based on available RAM

### Stage 3: Projection & Shingling
- **PCA projection**: Reduce to configurable dimension (default 256D)
- **Window shingling**: Sliding windows with positional encoding
- **Strand awareness**: Handle forward/reverse gene orientations

### Stage 4: Dual Indexing
- **Discrete**: MinHash LSH on protein sequence codewords
- **Continuous**: Signed random projection for embedding similarity

### Stage 5: Syntenic Block Discovery
- **Anchor identification**: High-similarity protein pairs
- **Collinear chaining**: Dynamic programming with gap penalties
- **DTW refinement**: Optional dynamic time warping alignment

### Stage 6: Block Clustering — Mutual-k Jaccard over Order-Aware Shingles

We cluster syntenic blocks into cassette families using **Mutual-k Jaccard** over order-aware shingles derived from SRP tokens of the existing window embeddings.

**Why:** Clustering on coarse scalars (e.g., DBSCAN on length/identity) collapses unrelated blocks into a single mode. We need both *content* and *order*, but without heavy all-vs-all alignment.

**Sketch → tokens:** For each window's embedding x, we compute a 256-bit SRP sign sketch (fixed seed), split into 32 bands of 8 bits, and hash each band to a 64-bit token. Similar windows share band tokens with high probability.

**Order-aware shingles:** Per block, we derive one stable token per window, then form k-gram shingles (default k=3) over the ordered token sequence and hash these into a set S_b. For strand −, we reverse order before shingling (internal only).

**Robustness gate (per block):** keep blocks with:
- alignment_length ≥ 4; spans ≥ 8 genes on both genomes; diagonal purity MAD(v=q_idx−t_idx) ≤ 1. Others go to sink.

**Graph construction:** Build an inverted index shingle→blocks; drop hub shingles with df>200. For each block b, enumerate candidates via postings; compute Jaccard J(S_b,S_c); keep top-k (k=3). Undirected edge b--c exists iff mutual-k holds and J ≥ 0.5.

**Clusters & labels:** Connected components of this sparse graph are clusters. Robust singletons (degree 0) are sent to the sink by default. Labels are deterministic:
- **0 = sink** (non-robust + singletons)
- 1..K assigned by decreasing component size, then increasing representative block id.

**Configuration:** see `analyze.clustering.*` in params (SRP bits/bands, k, jaccard_tau, mutual_k, df_max, robustness thresholds, sink semantics).

**Determinism:** All randomness is seeded; ordering keys are fixed; re-runs produce identical cluster IDs.

## Comprehensive Analysis

To run the complete analysis pipeline:

```bash
elsa analyze --output-dir results/
```

This performs all-vs-all locus comparison, finds syntenic blocks, clusters them with the new method, and outputs:
- `syntenic_blocks.csv` - All discovered blocks with cluster assignments
- `syntenic_clusters.csv` - Cluster summaries and statistics

## Configuration

Key parameters in `elsa.config.yaml`:

```yaml
plm:
  model: esm2_t33        # esm2_t33, esm2_t12, prot_t5
  device: auto           # auto, mps, cuda, cpu
  project_to_D: 256      # PCA target dimension
  
ingest:
  gene_caller: prodigal  # prodigal, metaprodigal, none
  min_cds_aa: 60         # Minimum protein length
  
discrete:
  K: 4096               # Codebook centroids
  minhash_hashes: 192   # MinHash functions

analyze:
  clustering:
    method: mutual_jaccard    # mutual_jaccard, dbscan, disabled
    sink_label: 0             # Sink cluster ID
    keep_singletons: false    # Keep robust singletons vs send to sink
    
    # Robustness gate
    min_anchors: 4            # Min alignment length
    min_span_genes: 8         # Min span on both genomes
    v_mad_max_genes: 1        # Max diagonal offset MAD
    
    # SRP tokenization
    srp_bits: 256             # Total projection bits
    srp_bands: 32             # Number of bands  
    srp_band_bits: 8          # Bits per band
    srp_seed: 1337            # Deterministic seed
    shingle_k: 3              # k-gram shingle size
    
    # Graph construction
    jaccard_tau: 0.5          # Min Jaccard similarity
    mutual_k: 3               # Mutual top-k parameter
    df_max: 200               # Max document frequency
  
system:
  rng_seed: 17          # Reproducibility seed
```

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only
- **Recommended**: 32GB RAM, GPU (MPS/CUDA)
- **Optimal**: 48GB RAM, Apple M4 Max or RTX 4090

## Testing

Use the provided test genomes:

```bash
# Download S. pneumoniae test data
wget https://example.com/spneumoniae_strain1.fasta
wget https://example.com/spneumoniae_strain2.fasta

# Run pipeline
elsa embed spneumoniae_strain*.fasta
elsa build
elsa find "strain1_contig1_10000:20000"
```

## API Usage

```python
from elsa import ELSAConfig, ProteinEmbedder, ProteinIngester

# Load configuration
config = ELSAConfig.from_yaml("elsa.config.yaml")

# Process genomes
ingester = ProteinIngester(config.ingest)
proteins = ingester.ingest_sample("genome.fasta", "sample1")

# Generate embeddings
embedder = ProteinEmbedder(config.plm)
embeddings = list(embedder.embed_sequences(proteins))
```

## Citation

```bibtex
@software{elsa2024,
  title={ELSA: Embedding Locus Shingle Alignment},
  author={Claude and Jacob},
  year={2024},
  url={https://github.com/user/elsa}
}
```