# ELSA: Embedding Locus Search and Alignment

Discover conserved gene neighborhoods (syntenic blocks) from protein language model embeddings.

ELSA uses gene-level anchor chaining to find variable-length syntenic blocks across assembled genomes or metagenomes. It achieves **98.9% operon recall** on experimentally verified E. coli operons and **97.1% ortholog precision** validated against OrthoFinder.

## Quick Start

```bash
# Install
pip install -e .

# Initialize config
elsa init

# Embed genomes (calls genes with Prodigal, embeds with ESM2)
elsa embed data/genomes/

# Find syntenic blocks
elsa analyze -c elsa.config.yaml -o syntenic_analysis \
    --genome-browser-db genome_browser/genome_browser.db \
    --sequences-dir data/genomes \
    --proteins-dir data/proteins

# Search for a specific locus
elsa search GCF_000005845:NC_000913:10-25 -c elsa.config.yaml

# Launch genome browser
cd genome_browser && streamlit run app.py
```

## How It Works

1. **Ingest** — Parse FASTA, call genes with Prodigal, translate to amino acids
2. **Embed** — Generate protein embeddings with ESM2 (or ProtT5, gLM2), L2-normalize
3. **Seed** — HNSW kNN search to find cross-genome gene anchors above a cosine similarity threshold
4. **Chain** — LIS-based collinear anchor chaining per contig pair with strand-aware partitioning
5. **Extract** — Greedy non-overlapping block selection by chain score
6. **Cluster** — Overlap-based grouping with genome support filtering

## Configuration

`elsa.config.yaml` — five sections:

```yaml
data:
  work_dir: ./elsa_index

ingest:
  gene_caller: prodigal
  min_cds_aa: 60

plm:
  model: esm2_t12          # esm2_t12, esm2_t33, prot_t5, glm2
  device: auto
  project_to_D: 0           # 0 = raw embeddings (recommended)
  l2_normalize: true

chain:
  similarity_threshold: 0.85
  max_gap_genes: 2
  min_chain_size: 2
  gap_penalty_scale: 0.0    # >0 enables concave gap penalty
  jaccard_tau: 0.3
  min_genome_support: 2

system:
  rng_seed: 17
```

## Output Files

- `micro_chain_blocks.csv` — All syntenic blocks with coordinates and scores
- `micro_chain_clusters.csv` — Cluster assignments and metadata

## Python API

```python
from pathlib import Path
from elsa.analyze.pipeline import run_chain_pipeline, ChainConfig

config = ChainConfig(
    hnsw_k=50,
    similarity_threshold=0.85,
    max_gap_genes=2,
    min_chain_size=2,
)

summary = run_chain_pipeline(
    genes_parquet=Path("elsa_index/ingest/genes.parquet"),
    output_dir=Path("syntenic_analysis/micro_chain"),
    config=config,
)
print(f"Blocks: {summary.num_blocks}, Clusters: {summary.num_clusters}")
```

## Cross-Species Datasets

For comparing genomes from different species, embed all genomes together:

```bash
# Put all genomes in one directory and embed together
elsa embed data/all_genomes/ -c cross_species.config.yaml
elsa analyze -c cross_species.config.yaml -o cross_species_results
```

Raw embeddings (`project_to_D: 0`) are recommended — they eliminate PCA fitting and produce equivalent results.

## Benchmark Results

Validated on 30-genome Enterobacteriaceae dataset (21 E. coli, 5 Salmonella, 4 Klebsiella):

| Metric | ELSA | MCScanX |
|--------|------|---------|
| Strict operon recall | 98.9% | 80.3% |
| Independent recall | 99.0% | 96.4% |
| Any coverage | 99.3% | 100.0% |
| Total blocks | 80,225 | 27,372 |
| Cross-genus blocks | 55,898 | 14,186 |
| Ortholog precision | 97.1% | — |
| Search recall @50 | 99.9% | — |

**Model concordance**: ESM2 and ProtT5 produce nearly identical results (within 1% recall, 5% block count), confirming the approach is PLM-agnostic.

## Key Source Files

| File | Role |
|------|------|
| `elsa/cli.py` | CLI interface (Click): embed, analyze, search, stats |
| `elsa/analyze/pipeline.py` | Batch pipeline orchestrator |
| `elsa/chain.py` | LIS-based collinear chaining + non-overlapping extraction |
| `elsa/seed.py` | GeneAnchor, cross-genome anchor discovery |
| `elsa/cluster.py` | Overlap-based block clustering |
| `elsa/search.py` | Locus-level search |
| `elsa/index.py` | HNSW / FAISS index building |
| `elsa/params.py` | Config validation (Pydantic) |
| `elsa/embeddings.py` | PLM model loading (ESM2, ProtT5, gLM2) |
| `elsa/ingest.py` | Genome parsing, gene calling |

## Hardware

- **Minimum**: 16GB RAM, CPU
- **Recommended**: 48GB unified memory, Apple M4 Max (MPS) or NVIDIA GPU (CUDA)

## Citation

```bibtex
@software{elsa2026,
  title={ELSA: Embedding Locus Search and Alignment},
  author={West-Roberts, Jacob},
  year={2026},
}
```
