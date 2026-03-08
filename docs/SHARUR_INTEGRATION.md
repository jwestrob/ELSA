# ELSA Synteny Discovery — Sharur Integration Guide

This document describes how to call ELSA's synteny pipeline from Sharur or any external data source. The primary interface is the `elsa synteny` CLI command, which requires **no ELSA config file**.

---

## Prerequisites

```bash
# ELSA must be installed in the same conda env (editable install)
pip install -e /path/to/ELSA

# Required: faiss-cpu, h5py, duckdb (for DuckDB path)
pip install faiss-cpu h5py duckdb
```

---

## Data Contract

ELSA needs two inputs: **protein metadata** and **protein embeddings**. They are joined on `protein_id` (called `gene_id` internally).

### 1. Protein Metadata — DuckDB (`--db`)

ELSA reads directly from a Sharur DuckDB `proteins` table:

```sql
SELECT protein_id, contig_id, bin_id, start, end_coord, strand
FROM proteins
ORDER BY bin_id, contig_id, start
```

| Sharur column | ELSA column | Type | Notes |
|---------------|-------------|------|-------|
| `protein_id` | `gene_id` | string | Unique per protein. **Must match HDF5 `protein_ids`.** |
| `bin_id` | `sample_id` | string | Genome / MAG identifier |
| `contig_id` | `contig_id` | string | Contig or scaffold ID |
| `start` | `start` | int | Genomic start coordinate (bp) |
| `end_coord` | `end` | int | Genomic end coordinate (bp) |
| `strand` | `strand` | `"+"`/`"-"` → `1`/`-1` | Automatically converted |

### 2. Protein Embeddings — HDF5 (`--embeddings`)

Standard Sharur format:

```
protein_embeddings.h5
├── protein_ids   # string array, shape (N,)
└── embeddings    # float32 matrix, shape (N, D)
```

- **Any embedding dimension works.** ELSA auto-detects D from the matrix shape.
- **Any PLM model works** (ESM2, ProtT5, gLM2, etc.) — the chaining algorithm is model-agnostic.
- Embeddings do NOT need to be L2-normalized; ELSA normalizes on load (disable with `--no-normalize`).
- **protein_ids must match the DuckDB `protein_id` column exactly.** Mismatches are silently dropped (logged to stderr).

### Alternative: Parquet Embeddings (`--embeddings-parquet`)

A parquet file with columns: `protein_id` (or `gene_id` or `id`) + `emb_000`, `emb_001`, ..., `emb_NNN`.

### 3. Annotations — DuckDB (`--annotations-db`)

ELSA reads from the Sharur `annotations` table to populate PFAM domains and a multi-source annotation table:

```sql
SELECT protein_id, source, accession, description, evalue, score
FROM annotations
```

| Column | Type | Notes |
|--------|------|-------|
| `protein_id` | string | Must match `proteins.protein_id` and HDF5 `protein_ids` |
| `source` | string | e.g., `'pfam'`, `'kofam'`, `'cazyme'`, `'defense_finder'` |
| `accession` | string | Domain/family ID (e.g., `'PF00005'`, `'K00001'`) |
| `description` | string | Human-readable name (optional) |
| `evalue` | float | E-value from HMM search (optional) |
| `score` | float | Bit score (optional) |

**Behavior:**
- Rows with `source = 'pfam'` are merged into `genes.pfam_domains` (semicolon-separated) — replaces Astra PFAM annotation
- ALL rows (every source) are bulk-loaded into `annotations_multi` table in the browser SQLite DB
- The `--annotations-db` can be the same DuckDB as `--db` if it has both `proteins` and `annotations` tables

---

## CLI Usage

### One-shot discovery (no persistent state)

```bash
elsa synteny \
    --db /path/to/sharur.duckdb \
    --embeddings /path/to/protein_embeddings.h5 \
    -o /path/to/results/
```

### With Sharur annotations (PFAM, KEGG, CAZy, DefenseFinder, etc.)

```bash
# Same DuckDB has both proteins + annotations tables
elsa synteny \
    --db /path/to/sharur.duckdb \
    --embeddings /path/to/protein_embeddings.h5 \
    --annotations-db /path/to/sharur.duckdb \
    -o /path/to/results/

# Or re-populate browser DB with annotations after synteny
elsa browser results/ \
    --store ./my_store \
    --annotations-db /path/to/sharur.duckdb
```

### With persistent FAISS store

The `--store` flag creates a persistent directory that saves the FAISS IVF-Flat index, metadata, and embeddings. Subsequent runs skip re-indexing.

```bash
# First run: creates the store + runs synteny
elsa synteny \
    --db /path/to/sharur.duckdb \
    --embeddings /path/to/protein_embeddings.h5 \
    --store /path/to/synteny_store \
    -o /path/to/results/

# Later: load store directly (no --db or --embeddings needed)
elsa synteny \
    --store /path/to/synteny_store \
    -o /path/to/results_v2/

# Add new genomes to existing store
elsa synteny \
    --store /path/to/synteny_store \
    --add-db /path/to/new_genomes.duckdb \
    --add-embeddings /path/to/new_embeddings.h5 \
    -o /path/to/results_v3/
```

### All parameters

```
--db PATH                     Sharur DuckDB path (protein metadata)
--proteins DIRECTORY          .faa FASTA files (alternative to --db)
--embeddings PATH             HDF5 embeddings (Sharur format)
--embeddings-parquet PATH     Parquet with emb_* columns
--store PATH                  Persistent FAISS store directory
--add-db PATH                 DuckDB with new genomes (requires --store)
--add-embeddings PATH         HDF5 with new embeddings (requires --store)
-o, --output-dir TEXT         Output directory [default: syntenic_output]
--similarity-threshold FLOAT  Cosine threshold for anchors [default: 0.85]
--max-gap INTEGER             Max gene gap in chains [default: 2]
--min-chain-size INTEGER      Min anchors per chain [default: 2]
--min-genome-support INTEGER  Min genomes per cluster [default: 2]
--gap-penalty-scale FLOAT     Concave gap penalty, 0=off [default: 0.0]
--jaccard-tau FLOAT           Cluster overlap threshold [default: 0.3]
--index-backend TEXT          auto/faiss/hnswlib [default: auto]
--hnsw-k INTEGER              k for neighbor search [default: 50]
--no-normalize                Skip L2 normalization
--annotations-db PATH         Sharur DuckDB with annotations table
                              (loads PFAM into gene table + all sources
                              into annotations_multi table; skips Astra)
```

---

## Output Files

All outputs land in `--output-dir`:

| File | Description |
|------|-------------|
| `micro_chain_blocks.csv` | One row per syntenic block. Columns: `block_id`, `cluster_id`, `query_genome`, `target_genome`, `query_contig`, `target_contig`, `query_start`/`end`, `target_start`/`end`, `n_anchors`, `chain_score`, `orientation`, `query_start_bp`/`end_bp`, `target_start_bp`/`end_bp`, `query_anchor_genes`, `target_anchor_genes` |
| `micro_chain_clusters.csv` | One row per cluster. Columns: `cluster_id`, `size`, `genome_support`, `mean_chain_length`, `genes_json` |
| `genome_browser.db` | SQLite DB for the genome browser UI |
| `schema/` | (if >=3 blocks/cluster) Structural architecture summaries |

### Browser DB tables (when `--annotations-db` is used)

| Table | Description |
|-------|-------------|
| `genes.pfam_domains` | Semicolon-separated PFAM accessions per gene (from `source='pfam'`) |
| `annotations_multi` | All annotation sources: `gene_id, source, accession, name, evalue, score` |

The `annotations_multi` table enables future UI features for KEGG, CAZy, DefenseFinder, etc. without schema changes.

### Interpreting blocks

Each block represents a **collinear run of homologous genes** between two contigs. Key fields:

- `n_anchors`: number of anchor gene pairs in the chain (2 = gene pair, 20+ = large operon)
- `chain_score`: sum of cosine similarities of anchor pairs
- `orientation`: `"same"` or `"inverted"` (relative strand)
- `query_anchor_genes` / `target_anchor_genes`: JSON arrays of gene IDs that are the actual anchors
- `*_start_bp` / `*_end_bp`: genomic coordinates bounding the block

### Interpreting clusters

Clusters group overlapping blocks across genome pairs into a single syntenic locus. A cluster with `genome_support=6` means all 6 genomes share that region.

---

## Persistent Store Layout

```
synteny_store/
├── index.faiss         # FAISS IVF-Flat index (inner product on L2-normed vectors)
├── metadata.parquet    # gene_id, sample_id, contig_id, start, end, strand
├── embeddings.npy      # (N, D) float32 matrix
└── config.json         # dim, n_vectors, nprobe, genomes list, emb_cols
```

The store is self-contained. To check what's in it:

```python
import json
config = json.loads(open("synteny_store/config.json").read())
print(config["n_vectors"], config["dim"], len(config["genomes"]), "genomes")
```

### Incremental add behavior

- `add_genes()` deduplicates by `gene_id` — already-indexed proteins are skipped
- The FAISS index is rebuilt from scratch on add (fast: ~2s for 100k vectors)
- The expensive artifacts (embeddings + metadata) are persisted, not recomputed
- New genomes appear in `config.json["genomes"]` after add

---

## Python API (for deeper integration)

```python
from pathlib import Path
from elsa.adapter import (
    load_proteins_from_duckdb,
    load_embeddings_h5,
    build_genes_dataframe,
)
from elsa.store import SyntenyStore
from elsa.analyze.pipeline import run_chain_pipeline, ChainConfig

# Load from Sharur
proteins = load_proteins_from_duckdb("/path/to/sharur.duckdb")
embeddings = load_embeddings_h5("/path/to/embeddings.h5")
genes_df = build_genes_dataframe(proteins, embeddings, normalize=True)

# Create or load store
store = SyntenyStore.create(Path("./my_store"), genes_df)
# store = SyntenyStore.load(Path("./my_store"))

# Run pipeline
summary = run_chain_pipeline(
    output_dir=Path("./results"),
    config=ChainConfig(),
    genes_df=store.get_genes_df(),
    prebuilt_index=store.get_index_tuple(),
)

print(f"blocks={summary.num_blocks}, clusters={summary.num_clusters}")
```

---

## Sharur HDF5 Writing Reference

For the Sharur agent building the embeddings file, here's what ELSA expects:

```python
import h5py
import numpy as np

# protein_ids: list of str, must match DuckDB protein_id column
# embeddings: np.ndarray, shape (N, D), float32

with h5py.File("protein_embeddings.h5", "w") as f:
    f.create_dataset("protein_ids", data=np.array(protein_ids, dtype="S"))
    f.create_dataset("embeddings", data=embeddings.astype(np.float32))
```

The `dtype="S"` creates fixed-length byte strings. ELSA handles both byte and unicode decoding.

---

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| "No protein IDs matched" | `protein_id` in DuckDB doesn't match `protein_ids` in HDF5 | Ensure exact string match between the two sources |
| OMP crash on macOS | Duplicate OpenMP libraries (faiss-cpu + system) | Set `KMP_DUPLICATE_LIB_OK=TRUE` in environment |
| 0 blocks found | All proteins from same genome, or embeddings not meaningful | Need >=2 genomes; verify embeddings aren't zeros |
| Very few blocks | `similarity_threshold` too high for divergent species | Try `--similarity-threshold 0.7` |
