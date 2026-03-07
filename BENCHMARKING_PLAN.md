# ELSA Benchmarking Plan

## Objective

Validate ELSA's syntenic block detection using:
1. **Primary**: Embedding-based conserved neighborhood ground truth
2. **Secondary**: PFAM domain overlap validation (where available)

## Key Principle

We're testing whether ELSA's complex pipeline (windowed embeddings → HNSW → Sinkhorn → clustering) recovers conserved gene neighborhoods that a simpler approach would identify.

---

## Approach

### Single-Species First

Start with within-species conservation before attempting cross-species analysis:
- 20 *E. coli* genomes
- 20 *B. subtilis* genomes
- 20 *S. pneumoniae* genomes

### Ground Truth Definition

A **conserved syntenic block** exists when:

1. Proteins P₁, P₂, ..., Pₙ are adjacent in genome A (≤2 intervening genes allowed)
2. Proteins Q₁, Q₂, ..., Qₙ are adjacent in genome B
3. For each pair: `cosine(embedding(Pᵢ), embedding(Qᵢ)) > τ` where τ = 0.9
4. Block must be present in **≥2 genomes**

### "Sameness" Criteria

| Method | Role | Threshold |
|--------|------|-----------|
| **pLM cosine similarity** | Primary | > 0.9 |
| **PFAM domain match** | Secondary validation | Exact match |

### Adjacency Rules

- **Max gap**: 2 intervening genes between block members
- **Min block size**: 3 genes
- **Strand**: Ignore for now (conserved blocks can flip)

---

## Data Requirements

### Genome Counts

| Species | Target | Source |
|---------|--------|--------|
| *Escherichia coli* | 20 | NCBI RefSeq |
| *Bacillus subtilis* | 20 | NCBI RefSeq |
| *Streptococcus pneumoniae* | 20 | NCBI RefSeq |

### Selection Criteria

- Complete genomes preferred (or high-quality drafts)
- Diverse strain selection (not all the same clone)
- RefSeq annotations available

---

## Pipeline

### Phase 1: Data Acquisition

```
1. Query NCBI for available genomes
2. Select 20 diverse strains per species
3. Download:
   - Genomic FASTA (.fna)
   - Protein FASTA (.faa)
   - Annotation (.gff)
4. Organize into benchmarks/data/{species}/
```

### Phase 2: ELSA Processing

```
1. Run ELSA embed on each species set
2. Run ELSA analyze (without --operon for now)
3. Output: syntenic_blocks.csv per species
```

### Phase 3: Ground Truth Generation

```
1. Load protein embeddings from ELSA output (genes.parquet)
2. For each genome pair:
   a. Compute pairwise protein cosine similarities
   b. Identify "same proteins" (cosine > 0.9)
   c. Find adjacent same-proteins in both genomes
   d. Record as ground truth block
3. Merge overlapping blocks across genome pairs
4. Output: ground_truth_blocks.tsv
```

### Phase 4: Evaluation

```
1. Load ELSA blocks and ground truth blocks
2. For each ground truth block:
   - Check if any ELSA block overlaps (≥50% Jaccard)
   - Record: hit/miss, overlap score, boundary accuracy
3. For each ELSA block:
   - Check if it matches any ground truth block
   - Record: true positive / false positive
4. Compute metrics
5. Secondary: Check PFAM consistency within blocks
```

---

## Evaluation Metrics

### Primary Metrics

| Metric | Definition |
|--------|------------|
| **Recall** | GT blocks overlapped by ELSA blocks / total GT blocks |
| **Precision** | ELSA blocks overlapping GT / total ELSA blocks |
| **F1** | Harmonic mean of recall and precision |
| **Mean Jaccard** | Average gene-set overlap for matched blocks |

### Secondary Metrics

| Metric | Definition |
|--------|------------|
| **PFAM coherence** | % of ELSA blocks where genes share PFAM domains |
| **Fragmentation** | GT blocks split across multiple ELSA blocks |
| **Over-merging** | Multiple GT blocks merged into one ELSA block |

### Overlap Definition

Two blocks "overlap" if:
- Jaccard(genes_A, genes_B) ≥ 0.5, OR
- genes_A ⊆ genes_B or genes_B ⊆ genes_A (containment)

---

## Directory Structure

```
benchmarks/
├── data/
│   ├── ecoli/
│   │   ├── genomes/           # .fna files
│   │   ├── proteins/          # .faa files
│   │   └── annotations/       # .gff files
│   ├── bacillus/
│   │   └── ...
│   └── spneumo/
│       └── ...
├── elsa_output/
│   ├── ecoli/
│   │   ├── elsa_index/
│   │   └── syntenic_analysis/
│   ├── bacillus/
│   └── spneumo/
├── ground_truth/
│   ├── ecoli_gt_blocks.tsv
│   ├── bacillus_gt_blocks.tsv
│   └── spneumo_gt_blocks.tsv
├── evaluation/
│   ├── ecoli_results.tsv
│   ├── bacillus_results.tsv
│   ├── spneumo_results.tsv
│   └── summary_report.md
└── scripts/
    ├── download_genomes.py
    ├── build_ground_truth.py
    └── evaluate_blocks.py
```

---

## Output Formats

### ground_truth_blocks.tsv

```
block_id    n_genes    n_genomes    genome_list    gene_positions
GT_0001     5          12           strain1,strain2,...    strain1:100-104,strain2:205-209,...
```

### evaluation_results.tsv

```
species     n_gt    n_elsa    recall    precision    f1    mean_jaccard    pfam_coherence
ecoli       342     289       0.85      0.91         0.88  0.72            0.68
bacillus    198     156       0.79      0.89         0.84  0.69            0.71
spneumo     167     143       0.82      0.87         0.84  0.70            0.65
```

---

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Cosine threshold (τ) | 0.9 | High similarity = same protein |
| Max adjacency gap | 2 genes | Allow small insertions/deletions |
| Min block size | 3 genes | Avoid trivial pairs |
| Min genomes | 2 | Conservation requires ≥2 instances |
| Overlap Jaccard | 0.5 | Standard overlap threshold |

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Recall | ≥ 0.80 | Catch most real conserved blocks |
| Precision | ≥ 0.70 | Majority of predictions are real |
| F1 | ≥ 0.75 | Balanced performance |

---

## Timeline

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| 1 | Download genomes | 1-2 hours |
| 2 | Run ELSA on 3 species | User-run (long) |
| 3 | Build ground truth | 30 min |
| 4 | Evaluation + report | 1 hour |

---

## Open Items

- [ ] Confirm genome selection after seeing what's available on NCBI
- [ ] Decide whether to use existing S. pneumo data or download fresh
- [ ] Set up PFAM annotations for secondary validation

---

## Issues Discovered During Benchmarking

### Performance Issues

- [ ] **`elsa analyze` needs progress bars** - Currently silent during "all-vs-all locus comparisons", no visibility into progress for long runs
- [ ] **O(n²) complexity in locus comparisons is brutal** - 20 E. coli genomes (90k windows, 35 loci, 595 comparisons) takes 20+ minutes vs seconds for 6-genome test set. Need to investigate:
  - Are we doing unnecessary pairwise window comparisons?
  - Can we use HNSW/LSH to prune candidate pairs before full comparison?
  - Is the chaining step the bottleneck?

### Ground Truth Issues

- [ ] **Union-find merging too aggressive** - Block 0 contains 4,326 genes (entire core genome) because ANY shared gene causes merge. Need:
  - Jaccard overlap threshold for merging, OR
  - Keep pairwise blocks as separate "instances", OR
  - Proper graph-based clustering with edge weights
