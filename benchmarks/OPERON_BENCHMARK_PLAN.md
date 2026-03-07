# Operon-Based Benchmark Plan for ELSA

## Overview

Benchmark ELSA's micro-synteny detection against experimentally verified operons in well-characterized prokaryotes. Operons are ideal ground truth because they represent functionally linked genes under evolutionary selection to remain syntenic.

## Target Organisms

### 1. Bacillus subtilis (Primary)
- **Why**: Already have 20 genomes indexed, well-curated operon database
- **Operon source**: DBTBS (Database of Transcriptional Regulation in B. subtilis)
- **Reference strain**: B. subtilis 168 (NC_000964)
- **Expected operons**: ~800 experimentally verified

### 2. Escherichia coli (Secondary)
- **Why**: Gold standard operon annotations, highest credibility for publication
- **Operon source**: RegulonDB
- **Reference strain**: E. coli K-12 MG1655 (NC_000913)
- **Expected operons**: ~950 experimentally verified
- **Genomes needed**: Download 15-20 diverse strains

---

## Phase 1: Download Operon Databases

### 1.1 DBTBS (B. subtilis)

**Source**: http://dbtbs.hgc.jp/

**Download approach**:
```bash
# DBTBS provides flat files - download operon list
mkdir -p benchmarks/operons/bsubtilis
cd benchmarks/operons/bsubtilis

# Option A: Direct download (if available)
wget http://dbtbs.hgc.jp/download/operon.txt

# Option B: Parse from web/API
# DBTBS has an API - may need to scrape or use their data dumps
```

**Alternative source**: SubtiWiki (http://subtiwiki.uni-goettingen.de/)
- More actively maintained
- Has downloadable operon tables
- Gene names map to B. subtilis 168

**Data format needed**:
```
operon_id    genes                      evidence
bsrA_operon  bsrA,bsrB,bsrC            experimental
dnaA_operon  dnaA,dnaN,recF,gyrB       experimental
...
```

### 1.2 RegulonDB (E. coli)

**Source**: https://regulondb.ccg.unam.mx/

**Download**:
```bash
mkdir -p benchmarks/operons/ecoli
cd benchmarks/operons/ecoli

# RegulonDB provides clean TSV downloads
# Go to: https://regulondb.ccg.unam.mx/menu/download/datasets/index.jsp
# Download: "Operon Set" (OperonSet.txt)

wget "https://regulondb.ccg.unam.mx/menu/download/datasets/files/OperonSet.txt" \
     -O regulondb_operons.txt

# Also download gene coordinates for mapping
wget "https://regulondb.ccg.unam.mx/menu/download/datasets/files/GeneProductSet.txt" \
     -O regulondb_genes.txt
```

**RegulonDB operon format**:
```
# Columns: Operon name, First gene, Last gene, Strand, Evidence, etc.
araBAD    araB    araD    forward    Strong
lacZYA    lacZ    lacA    forward    Strong
...
```

---

## Phase 2: Download Genomes

### 2.1 B. subtilis Genomes (Already Have)

Verify existing data:
```bash
# Check current B. subtilis genomes
ls -la data_bsubtilis/genomes/  # or wherever they are stored

# Should have ~20 genomes from previous benchmark
# Reference strain 168 must be included
```

If need to re-download or verify:
```bash
# NCBI datasets CLI
datasets download genome taxon "Bacillus subtilis" \
    --include genome,gff3,protein \
    --assembly-level complete \
    --limit 20 \
    --filename bsubtilis_genomes.zip
```

### 2.2 E. coli Genomes (Need to Download)

**Selection criteria**:
- Complete genomes only (no drafts)
- Diverse strains (K-12, O157:H7, various pathovars)
- Include reference K-12 MG1655

**Download script**:
```bash
mkdir -p data_ecoli/genomes data_ecoli/proteins

# Using NCBI datasets CLI
datasets download genome taxon "Escherichia coli" \
    --include genome,gff3,protein \
    --assembly-level complete \
    --reference \
    --limit 20 \
    --filename ecoli_genomes.zip

# Unzip and organize
unzip ecoli_genomes.zip -d ecoli_temp
# Move files to data_ecoli/genomes/ and data_ecoli/proteins/
```

**Recommended strains** (manually curate if needed):
| Strain | Accession | Notes |
|--------|-----------|-------|
| K-12 MG1655 | NC_000913 | Reference, RegulonDB annotations |
| K-12 W3110 | NC_007779 | Common lab strain |
| O157:H7 EDL933 | NC_002655 | Pathogenic |
| O157:H7 Sakai | NC_002695 | Pathogenic |
| BL21(DE3) | CP001509 | Expression strain |
| Nissle 1917 | CP007799 | Probiotic |
| CFT073 | NC_004431 | Uropathogenic |
| UTI89 | NC_007946 | Uropathogenic |
| IAI39 | NC_011750 | ExPEC |
| ED1a | NC_011745 | Commensal |
| ... | ... | Add 10 more diverse strains |

---

## Phase 3: Map Reference Operons to All Strains

### 3.1 The Challenge

Operon databases annotate a single reference strain. We need to:
1. Identify orthologs of reference operon genes in each target strain
2. Check if orthologs maintain synteny (same order, same strand)
3. An operon is "conserved" in strain X if all genes have syntenic orthologs

### 3.2 Ortholog Mapping Approach

**Option A: OrthoFinder (Recommended)**
```bash
# Already have OrthoFinder infrastructure from benchmarks/orthofinder/

# Run OrthoFinder on all genomes
orthofinder -f data_ecoli/proteins/ -t 8

# Output: Orthogroups/Orthogroups.tsv
# Maps each gene to its orthogroup across all genomes
```

**Option B: BLAST-based mapping**
```bash
# For each operon gene in reference:
# 1. BLAST against all target genomes
# 2. Take best hit with >80% identity, >80% coverage
# 3. Verify synteny of hits
```

### 3.3 Build Ground Truth File

**Script**: `benchmarks/scripts/build_operon_ground_truth.py`

```python
"""
For each reference operon:
1. Get gene list from operon DB
2. Find orthologs in each target genome via OrthoFinder
3. Check if orthologs are syntenic (consecutive positions, same strand)
4. Output: conserved operon instances across all genome pairs
"""

# Output format: ground_truth_operons.csv
# genome_a, genome_b, operon_id, genes_a, genes_b, conserved (bool)
```

**Ground truth criteria**:
- Operon is "detected" if ELSA block contains ≥N consecutive genes from operon
- N = min(3, operon_size - 1) for operons with 3+ genes
- For 2-gene operons: both genes must be in block

---

## Phase 4: Run ELSA Pipeline

### 4.1 B. subtilis

```bash
# Already have elsa_index for B. subtilis
# If not, run:
elsa init -c elsa_bsubtilis.config.yaml
elsa embed -c elsa_bsubtilis.config.yaml
elsa analyze -c elsa_bsubtilis.config.yaml

# Run micro-chain analysis
python -c "
from pathlib import Path
from elsa.analyze.micro_chain import run_micro_chain_pipeline, MicroChainConfig

config = MicroChainConfig(
    similarity_threshold=0.8,  # Same as Borg run
    hnsw_k=50,
    max_gap_genes=2,
    min_chain_size=2,
)

run_micro_chain_pipeline(
    genes_parquet=Path('elsa_index_bsubtilis/ingest/genes.parquet'),
    output_dir=Path('benchmarks/results/bsubtilis'),
    config=config,
)
"
```

### 4.2 E. coli

```bash
# Create config
cat > elsa_ecoli.config.yaml << 'EOF'
data:
  work_dir: ./elsa_index_ecoli
  allow_overwrite: false
plm:
  model: esm2_t12
  batch_amino_acids: 16000
  fp16: true
  project_to_D: 256
ingest:
  gene_caller: prodigal
  min_cds_aa: 60
EOF

# Run pipeline
elsa embed -c elsa_ecoli.config.yaml data_ecoli/genomes/*.fna
elsa analyze -c elsa_ecoli.config.yaml

# Run micro-chain
# (same as B. subtilis above, different paths)
```

---

## Phase 5: Evaluation

### 5.1 Metrics

**Primary metrics**:
- **Operon Recall**: fraction of conserved operons detected by ELSA
- **Operon Precision**: fraction of ELSA blocks that match operons
- **Boundary F1**: how well block boundaries match operon boundaries

**Detection criteria**:
```python
def operon_detected(operon_genes, block_genes, min_overlap=0.75):
    """Operon is detected if block contains ≥75% of operon genes in order."""
    operon_set = set(operon_genes)
    block_set = set(block_genes)
    overlap = len(operon_set & block_set)
    return overlap >= len(operon_genes) * min_overlap
```

### 5.2 Evaluation Script

**Script**: `benchmarks/scripts/evaluate_operon_benchmark.py`

```python
"""
Compare ELSA blocks to operon ground truth.

Inputs:
- ground_truth_operons.csv (from Phase 3)
- micro_chain_blocks.csv (from Phase 4)

Outputs:
- Per-operon detection status
- Aggregate metrics (recall, precision, F1)
- Confusion analysis (false positives, false negatives)
"""
```

### 5.3 Expected Results

Based on current B. subtilis benchmark (99.95% F1 on existing ground truth):

| Metric | Target | Notes |
|--------|--------|-------|
| Operon Recall | >90% | Detect most conserved operons |
| Operon Precision | >70% | Some blocks may span multiple operons |
| Boundary Accuracy | >80% | Block edges within 1-2 genes of operon edges |

---

## Phase 6: Analysis & Iteration

### 6.1 Error Analysis

For false negatives (missed operons):
- Are genes too divergent? (low embedding similarity)
- Is the operon split across contigs?
- Did the operon lose conservation in some strains?

For false positives (spurious blocks):
- Do they represent unannotated operons?
- Are they transposons/mobile elements?
- Domain-only similarity without functional linkage?

### 6.2 Parameter Sensitivity

Test different thresholds:
```python
for sim_threshold in [0.75, 0.80, 0.85, 0.90]:
    for max_gap in [1, 2, 3]:
        run_and_evaluate(sim_threshold, max_gap)
```

---

## File Structure

```
benchmarks/
├── OPERON_BENCHMARK_PLAN.md      # This file
├── operons/
│   ├── bsubtilis/
│   │   ├── dbtbs_operons.tsv     # Downloaded operon annotations
│   │   └── subtiwiki_operons.tsv # Alternative source
│   └── ecoli/
│       ├── regulondb_operons.txt # Downloaded from RegulonDB
│       └── regulondb_genes.txt   # Gene coordinates
├── scripts/
│   ├── download_operons.py       # Fetch operon databases
│   ├── download_ecoli_genomes.sh # NCBI genome download
│   ├── build_operon_ground_truth.py  # Map operons to all strains
│   └── evaluate_operon_benchmark.py  # Compare ELSA vs ground truth
└── results/
    ├── bsubtilis/
    │   ├── ground_truth_operons.csv
    │   ├── micro_chain_blocks.csv
    │   └── evaluation_report.json
    └── ecoli/
        ├── ground_truth_operons.csv
        ├── micro_chain_blocks.csv
        └── evaluation_report.json
```

---

## Execution Checklist

- [ ] **Phase 1**: Download operon databases
  - [ ] Download DBTBS / SubtiWiki operons for B. subtilis
  - [ ] Download RegulonDB operons for E. coli
  - [ ] Parse into standardized TSV format

- [ ] **Phase 2**: Prepare genomes
  - [ ] Verify B. subtilis genomes (should already have 20)
  - [ ] Download 15-20 E. coli complete genomes
  - [ ] Organize into data_ecoli/genomes/ and data_ecoli/proteins/

- [ ] **Phase 3**: Build ground truth
  - [ ] Run OrthoFinder on each organism set
  - [ ] Map reference operons to all strains via orthologs
  - [ ] Generate ground_truth_operons.csv for each organism

- [ ] **Phase 4**: Run ELSA
  - [ ] Run micro-chain pipeline on B. subtilis
  - [ ] Embed and run micro-chain on E. coli

- [ ] **Phase 5**: Evaluate
  - [ ] Run evaluation script
  - [ ] Generate metrics report
  - [ ] Analyze errors

- [ ] **Phase 6**: Iterate
  - [ ] Tune parameters based on error analysis
  - [ ] Re-run and compare

---

## Notes

1. **SubtiWiki vs DBTBS**: SubtiWiki is more actively maintained and may have better downloads. Try both.

2. **RegulonDB version**: Use latest stable release. Note version for reproducibility.

3. **Ortholog stringency**: Start strict (>90% identity), relax if recall is too low.

4. **Multi-copy genes**: Some operon genes have paralogs. Handle by requiring synteny, not just presence.

5. **Small operons**: 2-gene operons are common but hard to distinguish from random pairs. Consider minimum operon size of 3 for primary metrics.
