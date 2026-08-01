# Canonical Benchmark Results — No PCA (March 2026)

## Configuration

| Parameter | Value |
|-----------|-------|
| PLM | ESM2-t12 (35M params) |
| Embedding dim | 480 (raw, no PCA) |
| L2 normalize | Yes |
| Similarity threshold | 0.85 |
| max_gap_genes | 2 |
| gap_penalty_scale | 0.0 (hard cutoff) |
| min_chain_size | 2 |
| Index backend | FAISS IVF-Flat |
| Gene caller | Prodigal (single mode) |
| min_cds_aa | 60 |

## 1. E. coli Operon Recall (20 genomes, 90,589 genes)

Ground truth: 58 RegulonDB operons, 10,182 instances across 20 E. coli genomes.

| Metric | Value |
|--------|-------|
| **Strict recall** | **98.9%** (10,068/10,182) |
| **Independent recall** | **98.9%** (10,072/10,182) |
| **Any coverage** | **99.2%** (10,105/10,182) |
| Total blocks | 20,117 |
| Total clusters | 6,355 |
| Mean block size | 35.5 genes |
| Total anchors | 713,248 |

### Localization

Recall asks whether an operon lies inside some block, and a block spanning the
whole contig answers that perfectly. These figures say how much of the block is
the operon, and they qualify every recall number above. Reproduce with
`benchmarks/scripts/evaluate_operon_recall.py`, which now emits them on every
run; definitions in `elsa/localization.py`.

| Metric | Value |
|--------|-------|
| **Median boundary precision** | **0.0396** (95% CI 0.0330–0.0481) |
| Median pair IoU | 0.0397 |
| Median block span | 90 genes |
| Median locus span | 3 genes |
| Median block / locus | **30x** |
| Median containing excess | **174 genes** |

**98.9% of operon instances sit inside some block; that block is 4.0% operon by
extent.** Both statements describe the same run, and neither is complete alone.

This is a property of the cohort rather than of the similarity space. Blocks
here survive a 0.99 cosine cut and are unchanged by embedding post-processing
(`TRANSFORM_ABLATION.md`), because in 20 near-clonal genomes a 90-gene collinear
run is real — the chain passes through genuine orthologs in genuine order. A
locus boundary is visible where a cohort varies, so tightening it is a question
of cohort composition or of reporting sub-structure inside long blocks.

### Trimming a block to its conserved core does nothing

The natural next move — find the densely-anchored core inside a long block and
report that instead — was built and measured. Cores were extracted as the
contiguous window maximising `anchors − λ·length`, swept over λ ∈ {0.2 … 0.8},
and scored with the localization metric above.

| λ | recall | median boundary precision | median block span |
|---|---:|---:|---:|
| baseline (whole block) | 0.9888 | 0.0395 | 90 |
| 0.2 | 0.9888 | 0.0395 | 90 |
| 0.5 | 0.9886 | 0.0395 | 90 |
| 0.65 | 0.9881 | 0.0395 | 90 |
| 0.8 | 0.9755 | 0.0395 | 91 |

Flat, then it starts losing recall. The reason is structural: chaining runs with
`max_gap_genes=2`, so consecutive anchors sit at most two positions apart **by
construction** and a block cannot contain sparse padding.

Measured anchor density within blocks confirms it:

| Corpus | Median span | Median density | Blocks ≥90% anchored |
|---|---:|---:|---:|
| *E. coli* | 90 genes | **1.000** | 84.3% |
| DPANN | 3 genes | **1.000** | — |

Every position in the median block carries an anchor. There is no padding to
strip, because the whole block is genuinely supported — a real orthologous run
that happens to be much longer than the operon inside it. Core extraction is
dead on both ends: near-clonal blocks are uniformly dense, divergent blocks are
already 3 genes long.

The usable response is navigation rather than segmentation:
`GET /api/cluster/{key}/locate?q=…` reports where inside each locus a query sits
and what fraction of the locus it occupies — boundary precision measured against
the caller's own query, with no ground truth required.

### Which ground truth

`ecoli_operon_gt_v2.tsv` (10,182 pair rows) produces the numbers above.
`ecoli_operon_gt.tsv` (8,834 pair rows) scores **36.3% strict recall** on the
same blocks. The two files define loci differently; the audit records this under
"multiple incompatible runs"
(`docs/BENCHMARK_AUDIT_HANDOFF_2026-07-19.md`). Any recall figure from this
cohort is meaningless without naming its ground-truth file.

## 2. Cross-Species Analysis (30 genomes, 139,198 genes)

E. coli (20) + Salmonella (5) + Klebsiella (5).

### Operon Recall (E. coli subset)

| Metric | Value |
|--------|-------|
| **Strict recall** | **98.9%** (10,073/10,182) |
| **Independent recall** | **99.0%** (10,077/10,182) |
| **Any coverage** | **99.3%** (10,110/10,182) |

### Block Statistics

| Metric | Value |
|--------|-------|
| Total blocks | 80,225 |
| Total clusters | 33,969 |
| Mean block size | 18.3 genes |
| Total anchors | 1,466,629 |

### Cross-Genus Breakdown

| Genus Pair | Blocks | % |
|------------|--------|---|
| Escherichia-Klebsiella | 27,414 | 34.2% |
| Escherichia-Salmonella | 22,799 | 28.4% |
| Escherichia-Escherichia | 19,729 | 24.6% |
| Klebsiella-Salmonella | 7,322 | 9.1% |
| Klebsiella-Klebsiella | 2,283 | 2.8% |
| Salmonella-Salmonella | 678 | 0.8% |
| **Cross-genus total** | **57,535 (71.7%)** |

### Block Size Distribution (Cross-Species)

| Size Bucket | Count | % |
|-------------|-------|---|
| 2-5 genes | 33,756 | 42.1% |
| 6-10 | 13,683 | 17.1% |
| 11-25 | 16,901 | 21.1% |
| 26-50 | 9,310 | 11.6% |
| 51-100 | 4,661 | 5.8% |
| 101-500 | 1,879 | 2.3% |
| 500+ | 35 | 0.0% |

## 3. Runtime Profiling

| Dataset | Genes | Dim | Time (s) | Blocks |
|---------|-------|-----|----------|--------|
| E. coli (20 genomes) | 90,589 | 480 | ~300 | 20,117 |
| Cross-species (30 genomes) | 139,198 | 480 | ~610 | 80,225 |

Pipeline: FAISS IVF-Flat build + anchor seeding (k=50) + LIS chaining + overlap clustering.
Hardware: Apple M4 Max, 48GB unified memory.
Anchors discovered: 3,567,487 cross-genome anchors from 720 contig pairs.

## 4. Projection Ablation (Separate Gene Set)

Tested on 85,809 E. coli genes (from pre-computed .faa files, different from canonical).
All 8 projection techniques produce identical results (±0.3%).

| Technique | Independent Recall | Blocks |
|-----------|-------------------|--------|
| Raw 480D (L2 only) | 76.1% | 25,702 |
| Per-dataset PCA 256D | 76.1% | 25,730 |
| Frozen PCA 256D | 76.1% | 25,692 |
| PCA + whitening | 75.9% | 25,632 |
| All-but-top k=1 | 76.1% | 25,688 |
| All-but-top k=3 | 75.8% | 25,685 |
| FAISS OPQ | 76.1% | 25,686 |
| Whitened raw 480D | 76.0% | 25,689 |

Note: Lower recall vs canonical (76% vs 99%) due to .faa-based ingestion losing genomic coordinates. The relative comparison between techniques is valid.

## 5. Gap Penalty Comparison

All configs achieve equivalent recall. Concave penalty produces fewer, larger blocks.

| Config | Strict | Independent | Any | Blocks |
|--------|--------|-------------|-----|--------|
| Hard gap=2 (default) | 98.8% | 100.0% | 100.0% | 19,057 |
| Hard gap=5 | 98.5% | 100.0% | 100.0% | 11,694 |
| Concave scale=1.0 | 98.6% | 100.0% | 100.0% | 7,748 |

## 6. Search Recall@k (E. coli)

Query: each of 58 RegulonDB operons used as a search query (operon region + 2 gene flanks).
Target: all 90,589 genes in the FAISS IVF-Flat index.

| k | Mean Recall | Perfect Recall |
|---|-------------|----------------|
| 10 | 56.3% | 3/58 (5.2%) |
| **25** | **100.0%** | **58/58 (100%)** |
| **50** | **100.0%** | **58/58 (100%)** |

### Cross-Species Search Recall

Same 58 operons queried against the 30-genome (139,198 gene) cross-species index.

| k | Mean Recall | Perfect Recall |
|---|-------------|----------------|
| 10 | 55.4% | 3/58 (5.2%) |
| 25 | 99.4% | 51/58 (87.9%) |
| **50** | **100.0%** | **58/58 (100%)** |

Note: Recall@10 is bounded by having 19+ target genomes — finding all partners
in 10 results is mathematically constrained.

## 7. Block Size Distribution (E. coli)

| Size Bucket | Count | % |
|-------------|-------|---|
| 2-5 genes | 6,818 | 33.9% |
| 6-10 | 1,890 | 9.4% |
| 11-25 | 3,516 | 17.5% |
| 26-50 | 3,395 | 16.9% |
| 51-100 | 2,844 | 14.1% |
| 101-500 | 1,625 | 8.1% |
| 500+ | 29 | 0.1% |

Mean: 35.5, Median: 15.0, Max: 2,729 genes.
Forward orientation: 12,019 (59.7%), Reverse: 8,098 (40.3%).

## 8. MCScanX Comparison (Cross-Species, 30 genomes)

MCScanX run on the same 30 genomes with default parameters and DIAMOND blastp.
MCScanX GFF rebuilt from genes.parquet to use identical gene coordinates (v3 fixed).
Collinearity parsed with tab-split to correctly extract all gene pairs.

| Metric | ELSA | MCScanX | Ratio |
|--------|------|---------|-------|
| **Total blocks** | 80,225 | 28,196 | **2.85x** |
| **Cross-genus blocks** | 57,535 | 14,940 | **3.85x** |
| Cross-genus % | 71.7% | 53.0% | |
| E. coli-only blocks | 19,729 | 11,627 | 1.70x |
| **E. coli operon recall (strict)** | **98.9%** | **80.3%** | **1.23x** |
| E. coli operon recall (independent) | 99.0% | 96.4% | 1.03x |
| E. coli operon recall (any) | 99.3% | 100.0% | 1.00x |

### Species Pair Comparison

| Genus Pair | ELSA | MCScanX | Ratio |
|------------|------|---------|-------|
| Escherichia-Klebsiella | 27,414 | 9,037 | 3.03x |
| Escherichia-Salmonella | 22,799 | 4,430 | 5.15x |
| Klebsiella-Salmonella | 7,322 | 1,473 | 4.97x |
| Klebsiella-Klebsiella | 2,283 | 1,346 | 1.70x |
| Salmonella-Salmonella | 678 | 283 | 2.40x |

ELSA finds the most improvement over MCScanX for inter-genus comparisons,
especially Escherichia-Salmonella (5.15x) and Klebsiella-Salmonella (4.97x).

## 9. Orthogroup Validation

OrthoFinder (30 species, 137,965 proteins, DIAMOND ultra-sensitive, MCL clustering).
14,677 orthogroups assigned to 137,965 genes (99.1% coverage).

### Anchor-Level Ortholog Precision

For each genome pair, we re-extracted the actual chained anchor pairs (gene pairs
forming collinear chains) and checked if both genes share an orthogroup.

| Metric | Value |
|--------|-------|
| **Clean anchor precision** | **97.1%** (15,080/15,533) |
| All-anchor precision | 74.7% (29,050/38,879) |

"Clean" excludes anchors involving Prodigal-translated proteins with internal stop
codons (X residues), which DIAMOND cannot match but ESM2 embeddings still find.
~48% of Prodigal-called proteins contain X residues from internal stops.

### Per-Pair Breakdown (Top 10 pairs, clean anchors)

| Genome Pair | Precision | Anchors |
|-------------|-----------|---------|
| GCF_000931565 vs GCF_000987875 | 98.5% | 1,846 |
| GCF_000819645 vs GCF_000931565 | 97.9% | 1,912 |
| GCF_000784925 vs GCF_000967155 | 97.8% | 2,027 |
| GCF_000801185 vs GCF_000931565 | 97.9% | 1,874 |
| GCF_000801165 vs GCF_000931565 | 97.6% | 1,919 |
| GCF_000814145 vs GCF_000931565 | 97.2% | 1,860 |
| GCF_000931565 vs GCF_000967155 | 96.9% | 1,929 |
| GCF_000784925 vs GCF_000931565 | 96.2% | 2,039 |

## Known Limitations

1. **Internal stop codons**: Prodigal's `single` mode translates ~48% of CDS with
   internal stop codons (X residues). These produce valid ESM2 embeddings but are
   invisible to DIAMOND BLAST, creating a gap between embedding-based and
   sequence-based validation. Using `meta` mode or GFF-based gene calling would
   reduce this artifact.

2. **MCScanX comparison**: MCScanX uses OrthoFinder BLAST pairs (137,965/139,198 genes
   matched). The v3-fixed GFF maps MCScanX gene IDs to our parquet coordinates for
   consistent evaluation. Block-level counts (total, cross-genus) are from MCScanX's
   own output; operon recall uses our coordinate system.

## Files

| File | Description |
|------|-------------|
| `benchmarks/configs/ecoli_canonical.config.yaml` | E. coli config (no PCA) |
| `benchmarks/configs/cross_species_canonical.config.yaml` | Cross-species config (no PCA) |
| `benchmarks/elsa_output/ecoli/elsa_index/ingest/genes.parquet` | E. coli embeddings (90,589 x 480D) |
| `benchmarks/elsa_output/cross_species/ingest/genes.parquet` | Cross-species embeddings (139,198 x 480D) |
| `benchmarks/elsa_output/ecoli/syntenic_analysis/` | E. coli blocks and clusters |
| `benchmarks/elsa_output/cross_species/syntenic_analysis/` | Cross-species blocks and clusters |
| `benchmarks/ground_truth/ecoli_operon_gt_v2.tsv` | Operon ground truth |
| `benchmarks/ground_truth/orthogroups_canonical.tsv` | OrthoFinder gene->OG mapping (137,965 genes) |
| `benchmarks/orthofinder/canonical_input/` | OrthoFinder input + results |
| `benchmarks/evaluation/ecoli_block_statistics.json` | E. coli block stats |
| `benchmarks/evaluation/cross_species_block_statistics.json` | Cross-species block stats |
| `benchmarks/evaluation/search_recall_ecoli.csv` | Search recall@k results (E. coli) |
| `benchmarks/evaluation/search_recall_cross_species.csv` | Search recall@k results (cross-species) |
| `benchmarks/scripts/evaluate_search_recall.py` | Search recall benchmark script |
| `benchmarks/scripts/analyze_block_statistics.py` | Block statistics analysis script |
| `benchmarks/scripts/run_orthogroup_validation.py` | Orthogroup validation script |
| `benchmarks/results/mcscanx_comparison/cross_species_v3_fixed.gff` | Fixed MCScanX GFF (from genes.parquet) |
| `benchmarks/evaluation/mcscanx_operon_recall_v3_fixed.md` | Corrected MCScanX operon recall |
