# ELSA: Embedding Locus Shingle Alignment

Order-aware discovery of conserved gene neighborhoods (syntenic blocks) from protein embeddings.

What you get by default:
- Strand-insensitive tokens (we ignore strand in per-window SRP tokens)
- Order-aware shingles (k=3) with a skip-gram pattern [0,2,5]
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

This creates `elsa.config.yaml`. The repository ships a tuned default. Use it as-is unless you have a reason to change parameters.

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
- **ESM2** (t12, t33) and **ProtT5** supported
- **GPU optimization**: MPS on Apple Silicon, CUDA on Nvidia (auto-detected)
- **Memory management**: Adaptive batch sizing

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

### Stage 6: Block Clustering

<details>
<summary><strong>Mutual-Jaccard over Order-Aware Shingles (technical)</strong></summary>

We cluster syntenic blocks into cassette families using **Mutual-k Jaccard** over order-aware shingles derived from SRP tokens of the existing window embeddings.

Why: Clustering on coarse scalars (e.g., length/identity alone) collapses unrelated blocks. We need both content and order without heavy all-vs-all alignment.

Sketch → tokens: For each window embedding x, compute a 256-bit SRP sign sketch (fixed seed), split into 32 bands of 8 bits, and hash each band to a 64-bit token. Similar windows share band tokens with high probability.

Order-aware shingles: Per block, derive one token per window and form k-gram shingles (k=3, skip-gram). Tokens are strand-insensitive; order is preserved by the shingle sequence.

**Robustness gate (per block):** strict defaults
- alignment_length ≥ 5; spans ≥ 10 genes on both genomes; diagonal purity MAD(v=q_idx−t_idx) ≤ 1. Others go to sink.

Graph construction: Build an inverted index shingle→blocks; drop hub shingles with df>200. For each block b, enumerate candidates via postings; compute weighted Jaccard; require informative (low-DF) overlap.

Clusters & labels: Connected components of this sparse graph are clusters. Robust singletons (degree 0) go to sink by default. Labels are deterministic:
- **0 = sink** (non-robust + singletons)
- 1..K assigned by decreasing component size, then increasing representative block id.

Configuration: see `analyze.clustering.*` in params (SRP bits/bands, k, df_max, thresholds). Defaults are tuned; avoid changing unless necessary.

Determinism: All randomness is seeded; ordering keys are fixed; re-runs produce identical cluster IDs.

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

`elsa.config.yaml` is pre-tuned. Key aspects:
- Strict alignment (min_anchors=5; min_span_genes=10)
- Strand-insensitive SRP tokens; k=3 skip-gram [0,2,5]
- Weighted Jaccard with df_max=200
- Post-cluster attach enabled (safe thresholds)

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
    method: mutual_jaccard
    sink_label: 0
    keep_singletons: false
    # Robustness gate
    min_anchors: 5
    min_span_genes: 10
    v_mad_max_genes: 1
    # SRP tokenization
    srp_bits: 256
    srp_bands: 32
    srp_band_bits: 8
    srp_seed: 1337
    shingle_k: 3
    jaccard_tau: 0.5
    df_max: 200

system:
  rng_seed: 17
```

## Micro‑Synteny (2–3 gene cassettes)

An optional, sidecar module that detects conserved 2–3 gene cassettes as adjacency enrichment across genomes. It does not touch the long‑locus alignment or clustering pipeline and is disabled by default.

- CLI: `micro-synteny --genes-tsv genes.tsv [--gap-tolerance 1] [--min-genome-support 4] [--alpha 0.05] [--permutations 200] [--seed 17] [--loose-triads] [--output-dir ./micro_synteny_out]`
  - Or via the main CLI: `elsa micro-synteny --from-repo [flags]` (uses genome_browser DB + PFAM for OGs)
- Input TSV schema: columns `genome_id, contig_id, index, strand, og_id` (per‑gene positional order within contig, and orthogroup ID).
- Outputs (TSV in `--output-dir`):
  - `edges.tsv`: `OG_u, OG_v, support_genomes, expected_support, O_over_E, q_adj, modal_orientation, orientation_consistency, params_json`
  - `cassettes_pairs.tsv`: significant OG pairs with FDR `q_adj` and consensus orientation
  - `cassettes_triads.tsv`: 3‑gene cassettes as strict 3‑cliques (default) or loose wedges (with `--loose-triads`)
  - `instances_pairs.tsv`: per‑genome placements for significant pairs
  - `instances_triads.tsv`: per‑genome placements for triads

Statistics: adjacency significance is assessed by pooled permutations that shuffle OG labels within contigs per genome; p‑values are FDR‑controlled via Benjamini–Hochberg over pairs. Orientation evidence is summarized separately (binomial test) and reported but not used for FDR control.

Python API:

```python
from elsa.synteny import micro_synteny_call
edges, pairs, triads, inst_pairs, inst_triads = micro_synteny_call(
    genes_df, gap_tolerance=1, min_genome_support=4, permutations=200, seed=17
)
```

To drive this from in‑repo objects, implement `synteny/repo_adapter.py:get_genes_dataframe()` to produce the required DataFrame and run `micro-synteny --from-repo`.


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
from elsa.params import load_config
from elsa.ingest import ProteinIngester
from elsa.embeddings import ProteinEmbedder

cfg = load_config("elsa.config.yaml")

# Process one genome (FASTA)
ingester = ProteinIngester(cfg.ingest)
proteins = ingester.ingest_sample("data/genomes/GENOME.fna", sample_id="sample1")

# Generate embeddings
embedder = ProteinEmbedder(cfg.plm)
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
