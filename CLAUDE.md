# CLAUDE.md

---
## 🚀 CURRENT BEST: Gene-Level Anchor Chaining Pipeline (January 2026)

**This is our most successful micro-synteny approach to date.**

The gene-level anchor chaining pipeline (`elsa/analyze/micro_chain.py` + `elsa/analyze/gene_chain.py`) achieves:
- **82.6% independent recall** on E. coli operons (98.4% any coverage)
- **92% ortholog validation** - genes in blocks share orthogroups
- Variable-length, non-overlapping syntenic blocks (2-4000+ genes)
- Proper overlap-based clustering that groups related blocks across genome pairs

---
## 📋 NEXT STEPS (January 2026 Benchmarking)

### Completed
- [x] **Operon recall evaluation**: 82.6% independent recall on E. coli operons (98.4% any coverage)
- [x] Ortholog validation: 92.4% of genes in ELSA blocks are verified orthologs
- [x] Negative control: E. coli vs B. subtilis shows no spurious cross-phylum synteny
- [x] Downloaded Salmonella (5) + Klebsiella (5) genomes for cross-species benchmark
- [x] **Cross-species benchmark**: 30 Enterobacteriaceae genomes, 54,878 cross-genus blocks detected
- [x] **Cross-species ortholog validation**: 80%+ overlap for within-genus, 21% for E.coli↔others
- [x] **MCScanX comparison**: ELSA finds 2.7x more blocks, 3.7x more cross-genus
- [x] **gLM2 vs ESM2**: gLM2 150M achieves 83.1% operon recall (vs 82.6% ESM2), more within-species sensitivity
- [x] **Cryptic homology discovery**: 8,069 cross-genus blocks where ELSA finds synteny that BLAST misses (44% seq id → 0.97 emb sim)

### Cross-Species Results (COMPLETE - January 2026)

**Full Enterobacteriaceae dataset (30 genomes in unified PCA space):**
- 20 *E. coli* + 5 *Salmonella* + 5 *Klebsiella*
- 142,952 proteins embedded together
- ~40 minutes embedding time on M4 Max (MPS)

| Species Pair | Blocks | Mean Size | Max Size | Clusters |
|--------------|--------|-----------|----------|----------|
| E. coli ↔ E. coli | 19,256 | 37.2 genes | 2,793 | 19,210 |
| E. coli ↔ Salmonella | 21,663 | 13.4 genes | 100 | 21,609 |
| E. coli ↔ Klebsiella | 26,183 | 11.1 genes | 252 | 26,127 |
| Salmonella ↔ Salmonella | 694 | 57.2 genes | 937 | 694 |
| Klebsiella ↔ Klebsiella | 2,126 | 17.0 genes | 351 | 2,123 |
| Klebsiella ↔ Salmonella | 7,032 | 9.9 genes | 106 | 6,986 |

**Summary:**
| Metric | Value |
|--------|-------|
| Total syntenic blocks | 76,954 |
| Cross-genus blocks | 54,878 (71.3%) |
| Within-species blocks | 22,076 (28.7%) |
| Cross-genus mean size | 11.9 genes |
| Within-species mean size | 35.9 genes |
| Large blocks (>100 genes) | 1,856 |

**Key findings:**
- ✅ Cross-genus synteny works with unified PLM embeddings
- ✅ Block size gradient matches phylogenetic distance (within > cross)
- ✅ Salmonella strains are closely related (largest mean block size)
- ✅ 75 E.coli-Klebsiella blocks with >100 genes preserved

**Ortholog Validation (OrthoFinder):**

| Comparison | Blocks | Mean OG Overlap | ≥90% Overlap |
|------------|--------|-----------------|--------------|
| Salmonella ↔ Salmonella | 694 | **84.7%** | 443 (64%) |
| Klebsiella ↔ Klebsiella | 2,124 | **80.9%** | 1,059 (50%) |
| Klebsiella ↔ Salmonella | 7,026 | **80.9%** | 3,119 (44%) |
| E.coli ↔ E.coli | 15,029 | **59.7%** | 2,454 (16%) |
| E.coli ↔ Salmonella | 21,653 | 21.8% | 99 (0.5%) |
| E.coli ↔ Klebsiella | 26,138 | 20.2% | 200 (0.8%) |

**Interpretation:**
- Within-genus and Salmonella-Klebsiella blocks show high ortholog overlap (80%+)
- E.coli ↔ others have lower overlap due to strain-specific genes and HGT
- ELSA still finds conserved regions (21% orthogroups shared) even when gene content differs

Results: `benchmarks/results/cross_species_chain/micro_chain/`
Validation: `benchmarks/evaluation/cross_species_ortholog_validation.md`

### MCScanX Comparison (COMPLETE - January 2026)

| Metric | ELSA | MCScanX | ELSA Advantage |
|--------|------|---------|----------------|
| Total blocks | 76,954 | 28,196 | **2.7x more** |
| Cross-genus blocks | 54,878 | 14,940 | **3.7x more** |
| E.coli↔Salmonella | 21,663 | 4,430 | **4.9x more** |

**Key finding**: ELSA's PLM embeddings detect distant homology that BLAST misses, enabling 4-5x more cross-genus synteny detection.

**Cross-genus operon conservation (NEW)**:
- 100% of E. coli operons show cross-genus synteny in ELSA blocks
- 93% conserved in Salmonella (≥50% rate)
- 98% conserved in Klebsiella (≥50% rate)
- Essential operons (ATP synthase, ribosomal proteins) show 70-100% conservation

Report: `benchmarks/evaluation/cross_genus_operon_analysis.md`
Figures: `benchmarks/evaluation/figures/`

### Operon Recall Evaluation (COMPLETE - January 2026)

Evaluated ELSA and MCScanX against 58 E. coli operons from RegulonDB across 20 genomes (10,182 operon instances):

| Metric | ELSA | MCScanX | Winner |
|--------|------|---------|--------|
| Strict recall (raw) | 47.2% | 53.3% | — (see below) |
| **Strict recall (corrected)** | **47.2%** | **5.6%** | **ELSA (8.4x)** |
| **Independent recall** | **82.6%** | 55.3% | **ELSA (+27%)** |
| **Any coverage** | **98.4%** | 78.0% | **ELSA (+20%)** |

**CRITICAL FINDING: 89.5% of MCScanX "strict recall" are false positives.**

Deep analysis of all 5,425 MCScanX strict recall cases:

| Classification | Count | Percentage |
|----------------|-------|------------|
| **Accidental span (0% correspondence)** | **4,550** | **83.9%** |
| Weak correspondence (1-49%) | 308 | 5.7% |
| Partial correspondence (50-89%) | 440 | 8.1% |
| True correspondence (≥90%) | 127 | 2.3% |

When we require actual gene-to-gene correspondence (≥50% of operon genes map to each other):
- MCScanX adjusted strict recall: **5.6%** (down from 53.3%)
- ELSA strict recall remains: **47.2%**

**Why this happens**: MCScanX creates large blocks (mean 65 genes) that accidentally span small operons (mean 4 genes) without the operon genes being explicitly linked in the collinearity file.

**Conclusion**: ELSA definitively outperforms MCScanX on all metrics when corrected for accidental spans.

Reports:
- `benchmarks/evaluation/operon_correspondence_analysis.md` - **Deep analysis of MCScanX false positives**
- `benchmarks/evaluation/operon_recall_comparison.md` - Original comparison
- `benchmarks/evaluation/ELSA_vs_MCScanX_FULL_REPORT.md` - **Comprehensive comparison report**

### Optional Follow-ups
- [ ] Run OrthoFinder on B. subtilis to enable ortholog validation
- [ ] Test on more divergent species pairs (e.g., Pseudomonas)
- [ ] Compare to ntSynt (newer minimizer-based tool)

### gLM2 vs ESM2 Comparison (February 2026)

Tested gLM2 150M (genomic language model) against ESM2 650M on the same 30-genome Enterobacteriaceae dataset.

**Model Comparison:**
| Property | gLM2 150M | ESM2 650M |
|----------|-----------|-----------|
| Parameters | 150M | 650M |
| Training data | Genomic context | Protein sequences |
| Embedding time | ~3.5 hrs | ~40 min |
| Speed | 136 AA/sec | ~1000 AA/sec |

**Operon Recall (E. coli):**
| Metric | gLM2 150M | ESM2 650M | Difference |
|--------|-----------|-----------|------------|
| Strict recall | **49.0%** | 47.2% | +1.8% |
| Independent recall | **83.1%** | 82.6% | +0.5% |
| Any coverage | **98.7%** | 98.4% | +0.3% |

**Block Detection:**
| Metric | gLM2 150M | ESM2 650M | Notes |
|--------|-----------|-----------|-------|
| Total blocks | 78,519 | 76,954 | +2% |
| Clusters | 31,007 | 76,724 | 60% fewer |
| E.coli↔E.coli | 51,925 | 19,256 | **2.7x more** |
| E.coli↔Salmonella | 11,328 | 21,663 | 48% fewer |
| E.coli↔Klebsiella | 13,941 | 26,183 | 47% fewer |
| Cross-genus total | 25,269 | 47,846 | 47% fewer |
| Cross-genus mean size | 11.2 genes | 12.2 genes | Similar |

**Key Findings:**
- gLM2 achieves **slightly better operon recall** despite being 4.3x smaller
- gLM2 is **more sensitive to within-species variation** (2.7x more E.coli↔E.coli blocks)
- gLM2 is **more conservative across genera** (47% fewer cross-genus blocks)
- Genomic context training may improve precision at cost of cross-genus sensitivity
- gLM2 forms fewer, larger clusters (more block overlap)

**Interpretation:** gLM2's genomic context training appears to make it better at detecting fine-grained synteny within species while being more conservative about distant homology across genera. For operon-scale validation, both models perform similarly, but gLM2 may be preferable when precision matters more than cross-genus sensitivity.

Results: `benchmarks/results/cross_species_glm2/micro_chain/`
Evaluation: `benchmarks/evaluation/glm2_operon_recall.csv`
Config: `benchmarks/configs/cross_species_glm2.config.yaml`

### Cryptic Homology Discovery (February 2026)

ELSA's PLM embeddings detect synteny that BLAST/MCScanX miss due to sequence divergence.

**Case Study: Salmonella-E.coli ~100 kb syntenic region**

| Method | Genes Detected | Coverage | Identity/Similarity |
|--------|---------------|----------|---------------------|
| BLAST (MCScanX) | 11 | 11% | 44% sequence identity |
| ELSA (ESM2) | 97 | 95% | 0.97 embedding similarity |

**Key findings:**
- 8,069 cross-genus ELSA blocks with <10% MCScanX overlap
- Orthologous genes diverged below BLAST threshold (44% identity) but PLM embeddings preserve functional similarity (0.97)
- Core housekeeping genes (*icd*, *mnmA*, *phoP/Q*, *pot* operon) correctly matched
- Species-specific genes (*sifA* in Salmonella, *csg* in E.coli) show appropriately lower similarity (0.5-0.8)

Report: `benchmarks/evaluation/CRYPTIC_HOMOLOGY_ANALYSIS.md`
Figure: `benchmarks/evaluation/figures/cryptic_synteny_v2.png`

### Key Benchmark Files
| File | Description |
|------|-------------|
| `benchmarks/evaluation/ELSA_vs_MCScanX_FULL_REPORT.md` | **Comprehensive comparison report** |
| `benchmarks/evaluation/operon_correspondence_analysis.md` | **MCScanX false positive analysis (89.5%)** |
| `benchmarks/evaluation/operon_recall_comparison.md` | ELSA vs MCScanX operon recall |
| `benchmarks/evaluation/mcscanx_overprediction_analysis.md` | MCScanX block coherence analysis |
| `benchmarks/evaluation/fragmentation_analysis.md` | ELSA vs MCScanX block fragmentation |
| `benchmarks/evaluation/figures/` | Publication-quality comparison figures |
| `benchmarks/evaluation/BLOCK_VALIDATION_RESULTS.md` | E.coli ortholog validation (92%) |
| `benchmarks/evaluation/cross_species_ortholog_validation.md` | Cross-species OG validation |
| `benchmarks/CROSS_SPECIES_BENCHMARK_PLAN.md` | Cross-species plan |
| `benchmarks/configs/cross_species.config.yaml` | Config for 30-genome run (ESM2) |
| `benchmarks/configs/cross_species_glm2.config.yaml` | Config for 30-genome run (gLM2) |
| `benchmarks/evaluation/glm2_operon_recall.csv` | gLM2 operon recall evaluation |
| `benchmarks/results/cross_species_glm2/micro_chain/` | gLM2 blocks and clusters |
| `benchmarks/evaluation/CRYPTIC_HOMOLOGY_ANALYSIS.md` | **Cryptic homology case study** |
| `benchmarks/evaluation/figures/cryptic_synteny_v2.png` | Cryptic synteny figure |

### Analysis Scripts
| Script | Description |
|--------|-------------|
| `benchmarks/scripts/analyze_mcscanx_overprediction.py` | Analyze MCScanX block coherence |
| `benchmarks/scripts/analyze_operon_correspondence.py` | Check if operon genes correspond in MCScanX blocks |
| `benchmarks/scripts/analyze_fragmentation.py` | Compare ELSA/MCScanX block fragmentation |
| `benchmarks/scripts/create_comparison_figures.py` | Generate publication figures |

---

### Quick Start: Running the Chain Pipeline via CLI

```bash
# Run the full pipeline with --chain flag
elsa analyze -c elsa.config.yaml --chain -o syntenic_analysis \
    --genome-browser-db genome_browser/genome_browser.db \
    --sequences-dir data/genomes \
    --proteins-dir data/proteins

# Launch genome browser
cd genome_browser && streamlit run app.py
```

### Running on Multiple/Different Datasets

**CRITICAL: The manifest paths must match your work_dir!**

When copying an ELSA index to a new location or using a different dataset:

1. **Update config work_dir:**
   ```yaml
   # elsa_borg.config.yaml
   data:
     work_dir: ./elsa_index_borg  # Must match your index location
   ```

2. **Update MANIFEST.json paths:** The `--chain` pipeline reads gene paths from `MANIFEST.json`. If you copied an index, update all artifact paths:
   ```json
   // elsa_index_borg/MANIFEST.json
   "genes": {
     "path": "elsa_index_borg/ingest/genes.parquet",  // NOT "elsa_index/..."
     ...
   }
   ```

3. **Run with correct directories:**
   ```bash
   elsa analyze -c elsa_borg.config.yaml --chain \
       -o syntenic_analysis_borg \
       --genome-browser-db genome_browser/genome_browser_borg.db \
       --sequences-dir data_borg/genomes \
       --proteins-dir data_borg/proteins
   ```

### Cross-Species Comparison (Combining Datasets)

**PROBLEM:** Datasets embedded separately have different PCA projections, making
their embeddings incompatible. Cross-species synteny requires all genomes to
share the same embedding space.

**SOLUTION:** Use `--save-raw` to save unprojected embeddings, then combine and
project together:

```bash
# 1. Embed each dataset with --save-raw
elsa embed data/species_A/ --save-raw -c config_A.yaml
elsa embed data/species_B/ --save-raw -c config_B.yaml

# 2. Merge raw parquet files
python -c "
import pandas as pd
a = pd.read_parquet('elsa_index_A/ingest/genes_raw.parquet')
b = pd.read_parquet('elsa_index_B/ingest/genes_raw.parquet')
combined = pd.concat([a, b], ignore_index=True)
combined.to_parquet('combined_raw.parquet')
"

# 3. Project with unified PCA into a new work_dir
elsa project --raw combined_raw.parquet -c combined.config.yaml

# 4. Build index and analyze as usual
elsa build -c combined.config.yaml
elsa analyze -c combined.config.yaml --chain -o cross_species_analysis
```

**NOTE:** `elsa project` is only needed for this special case. Normal usage
should use `elsa embed` which handles PCA projection automatically.

### Python API (Alternative)

```python
from pathlib import Path
from elsa.analyze.micro_chain import run_micro_chain_pipeline, MicroChainConfig

config = MicroChainConfig(
    hnsw_k=50,
    similarity_threshold=0.9,
    max_gap_genes=2,
    min_chain_size=2,
)

summary = run_micro_chain_pipeline(
    genes_parquet=Path('elsa_index/ingest/genes.parquet'),
    output_dir=Path('syntenic_analysis/micro_chain'),
    config=config,
)
print(f'Blocks: {summary.num_blocks}, Clusters: {summary.num_clusters}')
```

### Key Files
- `elsa/analyze/gene_chain.py` - Core LIS-based chaining algorithm
- `elsa/analyze/micro_chain.py` - Pipeline orchestration and overlap-based clustering
- `elsa/analyze/micro_gene.py` - Legacy 3-gene window approach (deprecated)

### Why This Works
1. **Gene-level HNSW indexing** - Find similar genes across genomes via embedding similarity
2. **LIS-based chaining** - Extract collinear anchor chains using dynamic programming
3. **Non-overlapping extraction** - Greedy selection of highest-scoring chains
4. **Overlap-based clustering** - Blocks sharing genes (same genome, contig, position) cluster together

### Benchmark Results

#### Syntenic Block Detection
| Dataset | Genomes | Genes | Blocks | Clusters | Recall | Precision | F1 |
|---------|---------|-------|--------|----------|--------|-----------|-----|
| S. pneumoniae | 6 | 11,483 | 2,123 | 645 | 99.78% | 100% | 99.89% |
| **B. subtilis** | **20** | **79,680** | **9,194** | **3,070** | **99.98%** | **99.92%** | **99.95%** |
| Borg genomes | 15 | 12,710 | 1,901 | 1,882 | TBD | TBD | TBD |
| **Enterobacteriaceae** | **30** | **142,952** | **76,954** | **76,724** | N/A | N/A | N/A |

#### Operon-Based Validation (January 2026)
Validated against experimentally verified operons from SubtiWiki and RegulonDB:

| Organism | Unique Operons | Instances | Recall @50% | Recall @100% |
|----------|----------------|-----------|-------------|--------------|
| **B. subtilis** | 32 | 5,926 | **99.6%** | **99.6%** |
| **E. coli** | 58 | 10,182 | **98.9%** | **97.6%** |

Key finding: Recall is **stable across overlap thresholds** - when ELSA finds an operon, it finds the whole thing.

#### Ortholog Validation (E. coli)
ELSA blocks validated against OrthoFinder orthogroups:

| Metric | Value |
|--------|-------|
| Blocks validated | 19,279 |
| Mean ortholog fraction | **92.4%** |
| Median ortholog fraction | **98.0%** |
| Blocks with ≥90% orthologs | 80.8% |

This confirms ELSA finds true synteny (genes sharing orthogroups), not embedding artifacts.

#### Cross-Species Negative Control
E. coli vs B. subtilis (different phyla):
- Mean embedding similarity: **0.001** (random)
- Only 0.21% of gene pairs have >0.8 similarity
- High-similarity pairs are universal housekeeping genes only

ELSA does not hallucinate synteny between distant species.

---

⚠️  **RESOURCE CONTENTION**: This runs on M4 Max with **unified memory** (shared CPU/GPU RAM). **ONE COMPUTE JOB AT A TIME.** Do NOT run multiple jobs in parallel - this includes:
- Multiple `elsa embed` jobs
- `elsa embed` + OrthoFinder
- `elsa embed` + ground truth building (HNSW is CPU-intensive)
- Any combination of GPU and CPU-heavy tasks

Wait for one job to complete before starting another.

⚠️  **IMPORTANT**: Only let the human run long-running bash commands like `elsa embed` or `elsa build` - they will timeout. Short commands like `elsa analyze` are OK.

## Environment Setup

```bash
# Conda environment location
/Users/jacob/.pyenv/versions/miniconda3-latest/envs/elsa

# Activate before running any elsa commands
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
conda activate elsa

# Verify installation
which elsa  # Should show: /Users/jacob/.pyenv/versions/miniconda3-latest/envs/elsa/bin/elsa
```

**The `elsa` package is installed in editable mode from this repo.** Changes to code in `elsa/` are immediately reflected.

**Genome browser must run from the ELSA directory** (not a subdirectory) to properly import the elsa module:
```bash
cd /Users/jacob/Documents/Sandbox/elsa_test/ELSA/genome_browser && streamlit run app.py
```

---

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Available Datasets

| Dataset | Config | Work Dir | Data Dir | Description |
|---------|--------|----------|----------|-------------|
| S. pneumoniae (default) | `elsa.config.yaml` | `elsa_index/` | `data/` | 6 genomes, primary test set |
| Borg genomes | `elsa_borg.config.yaml` | `elsa_index_borg/` | `data_borg/` | 15 novel extrachromosomal elements |
| **E. coli** | `benchmarks/` | `benchmarks/elsa_output/ecoli/` | `benchmarks/data/ecoli/` | 20 genomes, operon benchmark |
| **B. subtilis** | `benchmarks/` | `benchmarks/elsa_output/bacillus/` | `benchmarks/data/bacillus/` | 20 genomes, operon benchmark |
| **Cross-species** | `benchmarks/configs/cross_species.config.yaml` | `benchmarks/data/cross_species/cross_species_index/` | `benchmarks/data/cross_species/` | 30 genomes (20 E.coli + 5 Salmonella + 5 Klebsiella) |
| **Cross-species (gLM2)** | `benchmarks/configs/cross_species_glm2.config.yaml` | `benchmarks/elsa_output/cross_species_glm2/` | `benchmarks/data/cross_species/` | Same 30 genomes, gLM2 150M embeddings |

### Benchmark Files
- Ground truth: `benchmarks/ground_truth/`
- Evaluation results: `benchmarks/evaluation/`
- ELSA blocks: `benchmarks/results/`

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

## Small‑Loci Adaptive Path (Implementation Notes)

Goal: improve recall for short loci (2–6 windows) without degrading purity for long/operon‑scale blocks. This path is off by default and can be toggled per‑run via config flags.

What’s implemented (code):
- File: `elsa/analyze/cluster_mutual_jaccard.py`
  - `enable_adaptive_shingles` (bool):
    - If true, shingling adapts by block length L:
      - L < 4 → k=2 contiguous (no skip)
      - 4 ≤ L < 6 → k=3 contiguous (no skip)
      - L ≥ 6 → configured default (k=3 with skip‑gram `[0,2,5]`)
  - `enable_small_path` (bool):
    - Relax auxiliary overlap checks when intersections are tiny and a “small” block is involved (still require `jaccard_tau`).
    - Require triangle support for edges touching small blocks:
      - `small_len_thresh` (int, default 6) — L < threshold is “small”
      - `small_edge_triangle_min` (int, default 1) — triangles required for small edges

Config profile to try:
- `configs/elsa_adaptive_small.yaml` — enables both paths and sets a looser source gate (min_anchors=3, min_span_genes=6), keeps default behavior for long loci.

How to run:
```bash
# Default (long/core clusters)
elsa analyze -c elsa.config.yaml

# Adaptive small-loci pass (keeps long behavior intact; adds sensitivity for short blocks)
elsa analyze -c configs/elsa_adaptive_small.yaml
```

Compare/merge strategy:
- Keep default clusters and IDs.
- Import the adaptive run; adopt only assignments for blocks that remained sink in the default run.
- Optional safety when merging: require small‑edge triangle support ≥2 or a minimum cluster size ≥3.

Purity mitigation knobs (recommended when enabling small path):
- Increase triangle support for small edges: set `small_edge_triangle_min: 2` (or 3) to prevent RP‑only pairs from latching onto canonical RP clusters.
- Tighten size compatibility for small→large edges (if you add this gate): require small edges to satisfy `size_ratio_min ≈ 0.7` for both shingle‑set size and alignment length.
- Informative‑overlap guard for tiny intersections (if you add this gate): when |intersection| ≤ 2 and a small block is involved, require `low_df_count ≥ 2` and `mean_idf ≥ 1.2`.
- Order coherence for short blocks (analysis gate): for L < 6, require perfect collinearity (v_mad == 0) or very small `v_mad_max_genes`.

Configs for small‑block exploration (optional):
- `configs/elsa_small_k2.yaml` — k=2 contiguous shingles; looser gate; mutual‑top‑k on.
- `configs/elsa_small_k1_strict.yaml` — k=1 (strict thresholds to compensate); mutual‑top‑k on.
- `configs/elsa_small_relaxed_gate.yaml` — only relaxes the source gate; keeps k=3 + skip‑gram.

Notes:
- Default run remains unchanged when flags are off.
- The adaptive small path focuses on edges involving short blocks; long→long behavior stays identical.

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

## TODOs — Micro‑Synteny Integration

1) Improve PFAM‑sparse consensus for small cassettes (PFAM → None → PFAM)

- Problem: Consensus currently uses per‑gene PFAM tokens. PFAM‑less genes break “consecutive proteins with shared domains”, so a true 2–3‑gene cassette can collapse to a single token in `cluster_consensus`.
- Plan (gap‑tolerant consensus):
  - Treat absent PFAM as a gap, not a break. When building per‑block token sequences in `compute_cluster_pfam_consensus`, skip empty tokens but allow adjacency over a small gap: max_gap=1 (i.e., use skip‑gram adjacency). This preserves PFAM₁→PFAM₂ adjacency across a PFAM‑less middle gene.
  - Relax global DF bans for micro clusters: for `cluster_type='micro_pair'`, bypass or raise `df_percentile_ban` so frequent tokens aren’t dropped in tiny clusters.
  - Lower minimum token requirements for micro clusters (e.g., no `min_token_per_block` enforcement) so 2‑token consensuses are retained.
  - Optional token backfill tiers for PFAM‑less genes (future):
    - Use orthogroup IDs when available (`ELSA_OG_COLUMN`) as fallback tokens.
    - As a last resort, assign lightweight “codeword” tokens from protein embeddings (KMeans codebook) to preserve cassette positional structure (down‑weighted in consensus).
  - UI: render a subtle “gap” marker between consensus tokens to signal PFAM‑less positions.

Implementation sketch:
- Add `allow_gaps: True`, `max_gap: 1` to `compute_cluster_pfam_consensus` pair construction; relax DF banning for `cluster_type='micro_pair'` (detected via a quick lookup) and skip any `min_token_per_block` gate.
- In `ELSADataIngester.compute_cluster_consensus`, pass micro‑friendly params when computing consensus for micro clusters.

2) Link small micro clusters to large clusters (future)

- Goal: If a micro cassette (2–3 genes) aligns at high identity within a larger long‑locus cluster, group them. This may deprecate the micro cluster in that view or show it as a “sub‑cassette”.
- Plan:
  - Add a `cluster_links` table with fields: `micro_cluster_id`, `parent_cluster_id`, `evidence_json` (overlap score, identity, shared PFAM pairs, coordinate containment), `status` (linked|deprecated).
  - Linking heuristic candidates:
    - Coordinate overlap: representative micro loci fall within the gene index span of representative long blocks on matching contigs.
    - PFAM consensus containment: micro consensus tokens appear as a contiguous subsequence in the long cluster’s consensus (gap‑tolerant).
    - Optional embedding evidence: high cosine between representative windows.
  - UI: show micro clusters under the parent cluster card; toggle to collapse “deprecated” micro clusters from the global list.

Notes:
- The default pipeline remains unchanged; these paths are opt‑in and micro‑specific. Consensus changes should not affect long‑locus clusters.
