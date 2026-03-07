# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

---
## ELSA v2: Gene-Level Anchor Chaining Pipeline

The gene-level anchor chaining pipeline achieves:
- **98.9% strict recall** on E. coli operons (99.2% any coverage)
- **97.1% ortholog precision** - chained anchor gene pairs share orthogroups
- **2.85x more blocks** than MCScanX, **3.85x more cross-genus**, **1.23x strict operon recall**
- **100% search recall@50** - all operon partners found in search results
- Variable-length, non-overlapping syntenic blocks (2-2700+ genes)

Full benchmark details: `benchmarks/evaluation/CANONICAL_BENCHMARKS_NOPCA.md`

---

## Warnings

**RESOURCE CONTENTION**: Runs on M4 Max with **unified memory**. **ONE COMPUTE JOB AT A TIME.** No parallel `elsa embed`, OrthoFinder, or HNSW-heavy tasks.

**LONG-RUNNING COMMANDS**: Use `run_in_background` for `elsa embed` and other GPU jobs. Short commands like `elsa analyze` can run inline.

---

## Environment Setup

```bash
# Conda environment
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
conda activate elsa
which elsa  # /Users/jacob/.pyenv/versions/miniconda3-latest/envs/elsa/bin/elsa
```

The `elsa` package is installed in **editable mode**. Changes to `elsa/` are immediately reflected.

Genome browser must run from the ELSA directory:
```bash
cd /Users/jacob/Documents/Sandbox/elsa_test/ELSA/genome_browser && streamlit run app.py
```

---

## Quick Start

```bash
# Run the full pipeline (chain is now the default and only mode)
elsa analyze -c elsa.config.yaml -o syntenic_analysis \
    --genome-browser-db genome_browser/genome_browser.db \
    --sequences-dir data/genomes \
    --proteins-dir data/proteins

# Search for syntenic regions matching a specific locus
elsa search GCF_000005845:NC_000913:10-25 -c elsa.config.yaml

# Launch genome browser
cd genome_browser && streamlit run app.py
```

### Running on Multiple/Different Datasets

**CRITICAL: The manifest paths must match your work_dir!**

When copying an ELSA index to a new location or using a different dataset:

1. **Update config work_dir:**
   ```yaml
   data:
     work_dir: ./elsa_index_borg  # Must match your index location
   ```

2. **Update MANIFEST.json paths:** The pipeline reads gene paths from `MANIFEST.json`. If you copied an index, update all artifact paths:
   ```json
   "genes": {
     "path": "elsa_index_borg/ingest/genes.parquet"
   }
   ```

3. **Run with correct directories:**
   ```bash
   elsa analyze -c elsa_borg.config.yaml \
       -o syntenic_analysis_borg \
       --genome-browser-db genome_browser/genome_browser_borg.db \
       --sequences-dir data_borg/genomes \
       --proteins-dir data_borg/proteins
   ```

### Cross-Species Comparison (Combining Datasets)

With raw embeddings (`project_to_D: 0`), no PCA alignment is needed. Just embed all genomes together:

```bash
# Put all genomes in one directory and embed together
elsa embed data/all_genomes/ -c cross_species.config.yaml

# Analyze
elsa analyze -c cross_species.config.yaml -o cross_species_analysis
```

### Python API

```python
from pathlib import Path
from elsa.analyze.pipeline import run_chain_pipeline, ChainConfig

config = ChainConfig(
    hnsw_k=50,
    similarity_threshold=0.85,
    max_gap_genes=2,
    min_chain_size=2,
    gap_penalty_scale=0.0,  # 0 = no gap penalty (legacy), >0 = concave penalty
)

summary = run_chain_pipeline(
    genes_parquet=Path('elsa_index/ingest/genes.parquet'),
    output_dir=Path('syntenic_analysis/micro_chain'),
    config=config,
)
print(f'Blocks: {summary.num_blocks}, Clusters: {summary.num_clusters}')
```

Backward-compatible imports still work:
```python
from elsa.analyze.micro_chain import run_micro_chain_pipeline, MicroChainConfig
```

---

## Available Datasets

| Dataset | Config | Work Dir | Data Dir | Description |
|---------|--------|----------|----------|-------------|
| S. pneumoniae (default) | `elsa.config.yaml` | `elsa_index/` | `data/` | 6 genomes, primary test set |
| Borg genomes | `elsa_borg.config.yaml` | `elsa_index_borg/` | `data_borg/` | 15 novel extrachromosomal elements |
| **E. coli** | `benchmarks/` | `benchmarks/elsa_output/ecoli/` | `benchmarks/data/ecoli/` | 20 genomes, operon benchmark |
| **B. subtilis** | `benchmarks/` | `benchmarks/elsa_output/bacillus/` | `benchmarks/data/bacillus/` | 20 genomes, operon benchmark |
| **Cross-species** | `benchmarks/configs/cross_species.config.yaml` | `benchmarks/data/cross_species/cross_species_index/` | `benchmarks/data/cross_species/` | 30 genomes (E.coli + Salmonella + Klebsiella) |
| **Cross-species (gLM2)** | `benchmarks/configs/cross_species_glm2.config.yaml` | `benchmarks/elsa_output/cross_species_glm2/` | `benchmarks/data/cross_species/` | Same 30 genomes, gLM2 150M embeddings |
| **Cross-species (ProtT5)** | `benchmarks/configs/cross_species_prott5.config.yaml` | `benchmarks/elsa_output/cross_species_prott5/` | `benchmarks/data/cross_species/` | Same 30 genomes, ProtT5-XL 1024D embeddings |

---

## Project Overview

ELSA (Embedding Locus Search and Alignment) discovers conserved gene neighborhoods (syntenic blocks) from protein language-model embeddings. It processes assembled genomes or metagenomes via collinearity chaining and clusters results via overlap-based grouping.

**Key Architecture:**
- **GPU-optimized** for M4 Max MacBook Pro (48GB unified memory) with CPU fallback
- Memory-mapped arrays for large datasets
- Gene-level HNSW indexing + LIS-based collinear chaining
- Concave gap penalties (minimap2-style, optional)
- Strand-aware anchor partitioning

## Key Source Files

| File | Lines | Role |
|------|-------|------|
| `elsa/cli.py` | ~660 | CLI interface (Click): embed, analyze, project, search, stats |
| `elsa/analyze/pipeline.py` | ~310 | Batch pipeline orchestrator |
| `elsa/chain.py` | ~280 | LIS-based collinear chaining + non-overlapping extraction |
| `elsa/seed.py` | ~160 | GeneAnchor, cross-genome anchor discovery |
| `elsa/cluster.py` | ~170 | Overlap-based block clustering |
| `elsa/search.py` | ~165 | Locus-level search |
| `elsa/index.py` | ~60 | HNSW index building |
| `elsa/params.py` | ~140 | Config validation (Pydantic) |
| `elsa/embeddings.py` | ~730 | PLM model loading (ESM2, ProtT5, gLM2) |
| `elsa/ingest.py` | ~540 | Genome parsing, gene calling |
| `elsa/projection.py` | ~430 | Projection (legacy PCA, rarely needed) |
| `elsa/manifest.py` | ~265 | Artifact manifest with hashes |
| `elsa/pfam_annotation.py` | ~325 | PFAM domain annotation |

**Backward-compat shims:** `elsa/analyze/gene_chain.py` and `elsa/analyze/micro_chain.py` re-export from new locations.

## Configuration System

- **Main config:** `elsa.config.yaml` — 5 sections: data, ingest, plm, system, chain
- **Work directory:** `./elsa_index/` — default location for all generated artifacts
- **Benchmark configs:** `benchmarks/configs/` — cross-species configs

Key config sections:
```yaml
data:
  work_dir: ./elsa_index
ingest:
  gene_caller: prodigal
  min_cds_aa: 60
plm:
  model: esm2_t12
  project_to_D: 0          # raw embeddings (recommended)
system:
  rng_seed: 17
chain:
  similarity_threshold: 0.85
  max_gap_genes: 2
  min_chain_size: 2
  gap_penalty_scale: 0.0  # >0 enables concave gap penalty
  jaccard_tau: 0.3
  min_genome_support: 2
```

Legacy config files (with `analyze.micro_chain.*` sections) are automatically mapped to the new `chain.*` section.

## Data Pipeline Architecture

**Stage sequence:**
1. **Ingest** — Parse FASTA, call genes (Prodigal/MetaProdigal), translate to AA
2. **PLM embeddings** — Load ESM2/ProtT5/gLM2, batch process proteins, L2 normalize
3. **Seed** — HNSW kNN to find cross-genome gene anchors
5. **Chain** — LIS-based collinear anchor chaining per contig pair
6. **Extract** — Greedy non-overlapping block selection
7. **Cluster** — Overlap-based grouping with genome support filtering

**Key file formats:**
- `genes.parquet` — projected protein embeddings
- `micro_chain_blocks.csv` — syntenic block output
- `micro_chain_clusters.csv` — cluster assignments

## Technical Constraints

- **Memory**: Memory-mapped arrays, float16 storage, batchable processing within 48GB envelope
- **GPU**: M4 Max MPS acceleration for PLM inference with automatic CPU fallback
- **Determinism**: All randomness seeded from `system.rng_seed`
- **Resumption**: Idempotent stages with checkpoints
- **Device support**: MPS (Metal Performance Shaders) primary, CUDA/CPU fallback

## TODOs

- [ ] Run OrthoFinder on B. subtilis for ortholog validation
- [ ] Test on more divergent species pairs (e.g., Pseudomonas)
- [ ] Compare to ntSynt (newer minimizer-based tool)
- [ ] Tune gap_penalty_scale on benchmark datasets
