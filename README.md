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