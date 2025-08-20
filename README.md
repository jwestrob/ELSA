# ELSA: Embedding Locus Shingle Alignment

Order-aware syntenic-block discovery from protein language-model embeddings.

Current default focuses on robust, compact clustering with:
- Strand-insensitive tokens (null strand-sign before SRP)
- Order-aware shingles with k=3 + skip-gram pattern [0,2,5]
- Strict alignment gate (min_anchors=5, min_span_genes=10)
- Safe, PFAM-agnostic post-cluster attach for tiny sink blocks

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

This creates `elsa.config.yaml`. The repository ships a tuned default (token-fix + attach). Use it as-is unless you have a reason to change parameters.

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

### 5. Run analysis pipeline

```bash
elsa analyze
```

Then launch the genome browser:

```bash
cd genome_browser
streamlit run app.py
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
- **Strand-insensitive tokens**: Before SRP tokenization, the strand-sign feature is nulled so per-window tokens match across forward/reverse loci; order-awareness is preserved via k-grams
- **Skip-gram shingles**: k=3 with [0,2,5] for robustness to small local reorders

### Stage 4: Dual Indexing
- **Discrete**: MinHash LSH on protein sequence codewords
- **Continuous**: Signed random projection for embedding similarity

### Stage 5: Syntenic Block Discovery
- **Anchor identification**: High-similarity protein pairs
- **Collinear chaining**: Dynamic programming with gap penalties
- **DTW refinement**: Optional dynamic time warping alignment

### Stage 6: Block Clustering (Mutual-Jaccard over shingles)

<details>
<summary><strong>Mutual-k Jaccard over Order-Aware Shingles</strong></summary>

We cluster syntenic blocks into cassette families using **Mutual-k Jaccard** over order-aware shingles derived from SRP tokens of the existing window embeddings.

**Why:** Clustering on coarse scalars (e.g., DBSCAN on length/identity) collapses unrelated blocks into a single mode. We need both *content* and *order*, but without heavy all-vs-all alignment.

**Sketch → tokens:** For each window's embedding x, we compute a 256-bit SRP sign sketch (fixed seed), split into 32 bands of 8 bits, and hash each band to a 64-bit token. Similar windows share band tokens with high probability.

**Order-aware shingles:** Per block, derive one token per window and form k-gram shingles (k=3, skip-gram). Tokens are strand-insensitive; order is preserved by the shingle sequence.

**Robustness gate (per block):** strict defaults
- alignment_length ≥ 5; spans ≥ 10 genes on both genomes; diagonal purity MAD(v=q_idx−t_idx) ≤ 1. Others go to sink.

**Graph construction:** Build an inverted index shingle→blocks; drop hub shingles with df>200. For each block b, enumerate candidates via postings; compute weighted Jaccard; require informative (low-DF) overlap.

**Clusters & labels:** Connected components of this sparse graph are clusters. Robust singletons (degree 0) go to sink by default. Labels are deterministic:
- **0 = sink** (non-robust + singletons)
- 1..K assigned by decreasing component size, then increasing representative block id.

**Configuration:** see `analyze.clustering.*` in params (SRP bits/bands, k, df_max, thresholds). Defaults are tuned; avoid changing unless necessary.

**Determinism:** All randomness is seeded; ordering keys are fixed; re-runs produce identical cluster IDs.

</details>

### Post-cluster Attach (sink rescue)

A narrow, PFAM-agnostic attach step safely promotes tiny sink blocks into existing clusters when strongly supported by union signatures (bandset and k=1), preserving compactness and purity.

## Comprehensive Analysis

To run the complete analysis pipeline:

```bash
elsa analyze --output-dir results/
```

This performs all-vs-all locus comparison, finds syntenic blocks, clusters them with the new method, and outputs:
- `syntenic_blocks.csv` - All discovered blocks with cluster assignments
- `syntenic_clusters.csv` - Cluster summaries and statistics

## Genome Browser

After running `elsa analyze`, launch the interactive genome browser to explore your results:

```bash
cd genome_browser
streamlit run app.py
```

The browser will be available at `http://localhost:8501` and provides:

### 🎯 Key Features

- **📊 Dashboard**: Overview statistics, size distributions, and genome comparison matrices
- **🔍 Block Explorer**: Advanced filtering (including Contig/PFAM substring filters), pagination, and detailed block information
- **🧬 Genome Viewer**: Interactive genome diagrams with gene arrows and domain tracks
- **🧩 Cluster Explorer**: Explore syntenic block clusters with functional analysis
- **🤖 AI Analysis**: GPT-powered functional interpretation of clusters

### 🔧 Navigation

1. **Dashboard Tab**: View overall statistics and data quality metrics
2. **Block Explorer Tab**: Filter and browse individual syntenic blocks
   - Use size sliders, identity thresholds, and genome selection
   - Search by PFAM domain substrings (comma-delimited) and/or contig substrings
   - Click blocks to view detailed genome context
3. **Genome Viewer Tab**: Visualize specific genomic loci
   - Enter locus coordinates (e.g., `genome1:100000-200000`)
   - View gene annotations with PFAM domain tracks
4. **Cluster Explorer Tab**: Analyze syntenic block clusters
   - Browse cluster overview cards
   - Click "Explore Cluster X" for detailed views
   - Generate AI functional analysis for each cluster

### 🧬 Genome Visualization

The genome diagrams show:
- **Gene arrows**: Strand-aware shapes colored by synteny role
- **PFAM domains**: Functional domain annotations below genes
- **Scale bars**: Genomic coordinates with smart labeling
- **Interactive elements**: Hover for gene details, zoom/pan navigation

### 🤖 AI-Powered Analysis

Each cluster can be analyzed using GPT-5 to provide:
- **Molecular mechanisms**: Specific pathways and enzyme functions
- **Conservation rationale**: Why these blocks are syntenic
- **Functional coherence**: Biological significance of co-localization

## Configuration (practical defaults)

`elsa.config.yaml` is pre-tuned for robust defaults. Key aspects:
- Strict alignment (min_anchors=5; min_span_genes=10)
- Strand-insensitive SRP tokens; k=3 skip-gram [0,2,5]
- Weighted Jaccard with df_max=200
- Post-cluster attach enabled (exp8 thresholds)

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

## Diagnostics

Useful scripts under `tools/`:
- `diagnose_block_vs_cluster.py`: explains why a specific block lacks edges to a target cluster (order modes, pre/post-DF overlap, bandset J)
- `evaluate_curated_rp_purity.py`: cluster compactness + curated RP purity
- `check_canonical_rp_alignment.py`: flags canonical RP loci aligned to non-canonical partners (should be 0)
- `attach_by_cluster_signatures.py`: PFAM-agnostic post-cluster attach

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