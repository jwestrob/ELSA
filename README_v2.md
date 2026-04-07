# ELSA: Embedding Locus Search and Alignment

ELSA discovers conserved gene neighborhoods (syntenic blocks) across bacterial and archaeal genomes using protein language model embeddings. Instead of requiring sequence homology searches like BLAST or DIAMOND, ELSA embeds each protein with ESM2, then finds collinear chains of similar genes across genomes — analogous to how minimap2 chains minimizer anchors, but at the gene level.

This makes it fast, alignment-free, and effective across genus-level divergence where sequence-based tools lose sensitivity. On a 30-genome Enterobacteriaceae benchmark, ELSA achieves **98.9% operon recall** and **97.1% ortholog precision**, finding **2.85x more syntenic blocks** than MCScanX.

## Prerequisites

- **Python 3.10+**
- **Prodigal** (gene caller) — must be on `$PATH`
- **Conda** (recommended) or pip

ELSA uses PyTorch for GPU-accelerated embedding. It runs on:
- Apple Silicon (MPS) — tested on M4 Max
- NVIDIA GPUs (CUDA)
- CPU (slower, but works)

## Installation

```bash
# Recommended: conda (installs Prodigal + all dependencies)
conda env create -f environment.yml
conda activate elsa
pip install -e .

# Alternative: pip only (you must install Prodigal separately)
pip install -e .
```

Verify the installation:

```bash
elsa --version
prodigal -v
```

## Quick Start

```bash
# 1. Initialize a config file
elsa init

# 2. Put your genome FASTA files (.fna, .fasta, .fa) in a directory
#    then embed them (calls genes with Prodigal, embeds with ESM2)
elsa embed data/genomes/ -c elsa.config.yaml

# 3. Find syntenic blocks
elsa analyze -c elsa.config.yaml -o results

# 4. Results are in results/micro_chain/
ls results/micro_chain/micro_chain_blocks.csv
ls results/micro_chain/micro_chain_clusters.csv
```

That's it. Steps 2-3 are the entire pipeline. Step 2 takes the longest (GPU embedding); step 3 is typically fast (seconds to minutes depending on dataset size).

### Shared cluster usage

If you're on a shared machine, use `--jobs` / `-j` to limit thread usage:

```bash
elsa embed data/genomes/ -c elsa.config.yaml -j 4
elsa analyze -c elsa.config.yaml -o results -j 4
```

## How It Works

1. **Ingest** — Parse FASTA genomes, call genes with Prodigal, translate to amino acid sequences
2. **Embed** — Generate protein embeddings with ESM2 (480-dimensional), L2-normalize
3. **Seed** — FAISS kNN search to find cross-genome gene pairs above a cosine similarity threshold
4. **Chain** — LIS-based collinear anchor chaining per contig pair, with strand-aware partitioning
5. **Extract** — Greedy non-overlapping block selection by chain score
6. **Cluster** — Overlap-based grouping of blocks into syntenic regions, with genome support filtering

## Configuration

`elsa init` generates a commented config file. The parameters you're most likely to tune:

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `chain.similarity_threshold` | 0.85 | Minimum cosine similarity between gene embeddings to call an anchor. Lower = more sensitive, more false positives. |
| `chain.max_gap_genes` | 2 | Maximum positional gap between anchors in a chain. Lower = stricter collinearity. |
| `chain.min_chain_size` | 2 | Minimum anchors to keep a chain as a block. |
| `chain.min_genome_support` | 2 | Minimum genomes a cluster must span to be reported. |
| `plm.model` | esm2_t12 | Embedding model. `esm2_t12` (fast, 480D) recommended; `esm2_t33` (1280D) is more accurate but slower. |
| `plm.device` | auto | `auto` detects GPU; set `cpu` to force CPU. |
| `ingest.gene_caller` | prodigal | `prodigal` for isolate genomes, `metaprodigal` for metagenomes. |

All other parameters (HNSW/FAISS tuning, gap penalties, clustering internals) have sensible defaults and are documented with comments in the generated config. See them with `elsa stats -c elsa.config.yaml`.

## Understanding the Output

### `micro_chain_blocks.csv`

Each row is a syntenic block — a collinear chain of anchor genes between two contigs.

| Column | Meaning |
|--------|---------|
| `block_id` | Unique block identifier |
| `query_genome` / `target_genome` | The two genomes this block connects |
| `query_contig` / `target_contig` | The specific contigs |
| `query_start` / `query_end` | Gene position range on the query contig |
| `target_start` / `target_end` | Gene position range on the target contig |
| `n_anchors` | Number of anchor gene pairs in the chain |
| `chain_score` | Sum of cosine similarities of all anchors (higher = stronger signal) |
| `orientation` | `1` = same strand, `-1` = inverted |

### `micro_chain_clusters.csv`

Blocks that share genomic regions are grouped into clusters. Each cluster represents a conserved syntenic region found across multiple genomes.

| Column | Meaning |
|--------|---------|
| `cluster_id` | Cluster identifier (0 = singleton / unclustered) |
| `size` | Number of blocks in this cluster |
| `genome_support` | Number of distinct genomes contributing blocks |
| `mean_chain_length` | Average number of anchors per block in the cluster |

A cluster with high `genome_support` and consistent `mean_chain_length` across its blocks is a well-conserved syntenic region (e.g., an operon or genomic island present in many genomes).

## Searching for Specific Loci

Once you've run the pipeline, you can search for syntenic blocks matching a specific genomic region:

```bash
# Search by position (genome:contig:start_gene-end_gene)
elsa search "GCF_000005845:NC_000913:10-25" -c elsa.config.yaml

# Search with a protein FASTA file (embeds on-the-fly)
elsa search query.faa -c elsa.config.yaml

# Search with a GFF + nucleotide FASTA
elsa search query.gff --fasta query.fna -c elsa.config.yaml
```

## Genome Browser

ELSA includes a Streamlit-based genome browser for visual exploration of syntenic blocks:

```bash
# Run analysis with genome browser database
elsa analyze -c elsa.config.yaml -o results \
    --genome-browser-db genome_browser/genome_browser.db \
    --sequences-dir data/genomes \
    --proteins-dir data/proteins

# Launch the browser
cd genome_browser && streamlit run app.py
```

## Cross-Species Comparison

To compare genomes from different species, put all genomes in one directory and process them together. No special configuration is needed — raw embeddings (the default `project_to_D: 0`) work across species without PCA alignment:

```bash
elsa embed data/all_genomes/ -c elsa.config.yaml
elsa analyze -c elsa.config.yaml -o cross_species_results
```

## Benchmark Results

Validated on a 30-genome Enterobacteriaceae dataset (21 *E. coli*, 5 *Salmonella*, 4 *Klebsiella*) against experimentally verified operons from RegulonDB and OrthoFinder ortholog groups:

| Metric | ELSA | MCScanX |
|--------|------|---------|
| Strict operon recall | 98.9% | 80.3% |
| Independent recall | 99.0% | 96.4% |
| Any coverage | 99.3% | 100.0% |
| Total blocks | 80,225 | 27,372 |
| Cross-genus blocks | 55,898 | 14,186 |
| Ortholog precision | 97.1% | -- |
| Search recall @50 | 99.9% | -- |

**Model concordance**: ESM2 and ProtT5 produce nearly identical results (within 1% recall, 5% block count), confirming the chaining algorithm dominates performance and the approach is PLM-agnostic.

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

## Troubleshooting

**`prodigal: command not found`** — Install via conda (`conda install -c bioconda prodigal`) or from [source](https://github.com/hyattpd/Prodigal).

**`elsa embed` is slow / using CPU** — Check `plm.device` in your config. If `auto` isn't detecting your GPU, set it explicitly to `mps` (Mac) or `cuda` (NVIDIA). Verify with: `python -c "import torch; print(torch.backends.mps.is_available())"` (Mac) or `python -c "import torch; print(torch.cuda.is_available())"` (Linux).

**Out of memory during embedding** — Lower `plm.batch_amino_acids` in your config (default 16000). Try 8000 or 4000.

**`genes.parquet not found`** — You need to run `elsa embed` before `elsa analyze`. The embed step produces `elsa_index/ingest/genes.parquet`.

**Config error: "Extra inputs are not permitted"** — You have a typo in your config file. The error message shows the misspelled key. Check spelling against `elsa init --force` output.

**FAISS OpenMP crash on macOS** — ELSA handles this internally (`KMP_DUPLICATE_LIB_OK`). If you still see issues, ensure you're not importing FAISS in your own code before importing ELSA.

## Key Source Files

| File | Role |
|------|------|
| `elsa/cli.py` | CLI interface (Click): embed, analyze, search, stats |
| `elsa/analyze/pipeline.py` | Batch pipeline orchestrator |
| `elsa/chain.py` | LIS-based collinear chaining + non-overlapping extraction |
| `elsa/seed.py` | Cross-genome anchor discovery via kNN |
| `elsa/cluster.py` | Overlap-based block clustering |
| `elsa/search.py` | Locus-level search |
| `elsa/index.py` | FAISS / HNSW index building |
| `elsa/params.py` | Config validation (Pydantic) |
| `elsa/embeddings.py` | PLM model loading (ESM2, ProtT5, gLM2) |
| `elsa/ingest.py` | Genome parsing, gene calling |

## Hardware

- **Minimum**: 16GB RAM, CPU, Prodigal on PATH
- **Recommended**: 32GB+ RAM, Apple M-series (MPS) or NVIDIA GPU (CUDA)
- **Tested at scale**: 3,564 genomes / 3.4M genes on M4 Max (48GB) in ~30 minutes

## Citation

```bibtex
@software{elsa2026,
  title={ELSA: Embedding Locus Search and Alignment},
  author={Westbrook, Jacob},
  year={2026},
}
```
