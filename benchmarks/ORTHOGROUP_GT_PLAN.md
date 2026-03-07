# Orthogroup-Based Ground Truth Plan

## Goal

Build ground truth "conserved gene neighborhoods" based on orthogroup co-occurrence patterns across multiple genomes. This directly measures what ELSA should find: genes that travel together as recombination units.

---

## Overview

```
Genes → Orthogroups → Ordered OG sequences → Conserved k-mers → GT Blocks
```

Instead of pairwise embedding similarity, we ask: "Which orthogroup combinations consistently appear adjacent across genomes?"

---

## Phase 1: Assign Genes to Orthogroups

**Input**: `genes.parquet` with embeddings for all genes across all genomes

**Method**:
1. Extract all gene embeddings (90k genes × 256 dims)
2. Cluster genes by embedding similarity using HDBSCAN or Leiden clustering
   - Genes in the same cluster = same orthogroup (OG)
   - Threshold: cosine similarity ~0.85-0.90 for same OG
3. Assign each gene an orthogroup ID: `OG_00001`, `OG_00002`, ...

**Output**: `orthogroups.tsv`
```
gene_id              sample_id        contig_id    position    orthogroup
GCF_000597845.1_1_1  GCF_000597845.1  NZ_CP007265  0           OG_00042
GCF_000597845.1_1_2  GCF_000597845.1  NZ_CP007265  1           OG_00108
...
```

**Validation**:
- How many orthogroups? (expect ~5-6k for E. coli core + accessory genome)
- Orthogroup size distribution (most should be ~20 for 20 genomes, some smaller for accessory genes)

---

## Phase 2: Build Orthogroup Sequences

**Input**: `orthogroups.tsv`

**Method**:
1. For each (sample_id, contig_id), sort genes by position
2. Extract ordered orthogroup sequence: `[OG_042, OG_108, OG_055, ...]`
3. Also record strand for each gene (for orientation-aware matching)

**Output**: `og_sequences.json`
```json
{
  "GCF_000597845.1:NZ_CP007265": {
    "orthogroups": ["OG_042", "OG_108", "OG_055", ...],
    "strands": [1, 1, -1, ...],
    "gene_ids": ["GCF_000597845.1_1_1", "GCF_000597845.1_1_2", ...]
  },
  ...
}
```

---

## Phase 3: Find Conserved Neighborhoods

**Input**: `og_sequences.json`

**Method**:
1. For each contig, extract all k-mers of orthogroups (k=3,4,5)
   - `[OG_042, OG_108, OG_055]` → one 3-mer
   - Canonicalize for strand: sort (forward, reverse-complement) lexicographically
2. Count genome support for each k-mer:
   - k-mer X appears in genomes {G1, G3, G5, G7, ...}
   - Support = number of genomes (not instances)
3. Filter to k-mers with support ≥ threshold (e.g., ≥50% of genomes, or ≥10 genomes)

**Output**: `conserved_kmers.tsv`
```
kmer_id    orthogroups              n_genomes    genome_list
KM_00001   OG_042,OG_108,OG_055     18           GCF_000597845.1,GCF_000599625.1,...
KM_00002   OG_108,OG_055,OG_201     17           GCF_000597845.1,GCF_000599645.1,...
...
```

---

## Phase 4: Merge into Conserved Blocks

**Input**: `conserved_kmers.tsv`, `og_sequences.json`

**Method**:
1. For each genome, find positions of conserved k-mers
2. Merge overlapping/adjacent k-mers into larger blocks
   - If KM_00001 and KM_00002 overlap by 2 OGs, merge them
3. A "conserved block" = maximal run of orthogroups that appears in ≥M genomes

**Output**: `gt_conserved_blocks.json`
```json
[
  {
    "block_id": "GT_0001",
    "orthogroups": ["OG_042", "OG_108", "OG_055", "OG_201"],
    "n_genomes": 17,
    "instances": {
      "GCF_000597845.1": {"contig": "NZ_CP007265", "genes": ["..._1", "..._2", "..._3", "..._4"]},
      "GCF_000599625.1": {"contig": "NZ_CP007390", "genes": ["..._5", "..._6", "..._7", "..._8"]},
      ...
    }
  },
  ...
]
```

---

## Phase 5: Evaluate ELSA Against GT

**Metrics**:

### 1. Orthogroup Recovery (Primary)
For each GT block's orthogroup set, does an ELSA cluster contain the same orthogroups?
- **Precision**: OGs in ELSA cluster that are in GT block
- **Recall**: OGs in GT block that are in ELSA cluster
- **F1**: Harmonic mean

### 2. Genome Coverage Agreement
For each GT block, compare genome sets:
- GT block appears in genomes {A, B, C, D}
- ELSA cluster spans genomes {A, B, C, E}
- Jaccard similarity of genome sets

### 3. Block Boundary Accuracy
Do ELSA block boundaries match GT block boundaries?
- Measure position offset of start/end genes

### 4. Co-clustering Accuracy (Adjusted Rand Index)
For all gene pairs:
- Are genes in the same GT block also in the same ELSA cluster?
- ARI measures agreement of the two clusterings

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `og_similarity_threshold` | 0.85 | Cosine threshold for same orthogroup |
| `og_clustering_method` | "leiden" | HDBSCAN, Leiden, or Louvain |
| `kmer_sizes` | [3, 4, 5] | k-mer sizes to consider |
| `min_genome_support` | 0.5 | Fraction of genomes required (or absolute number) |
| `merge_gap` | 1 | Max OG gap when merging k-mers |

---

## Expected Outcomes

### For E. coli (20 strains, ~95% similar):
- Most orthogroups should have 20 members (one per genome)
- Most of the chromosome should be one big conserved block (core genome)
- Variable regions (prophages, genomic islands) should be separate blocks with lower genome support

### What this tells us about ELSA:
- Does ELSA's clustering recover the same orthogroup neighborhoods?
- Does ELSA correctly identify which genomes share each block?
- Does ELSA find the boundaries of conserved regions?

---

## Implementation Order

1. `build_orthogroups.py` - Phase 1: Cluster genes into orthogroups
2. `build_og_sequences.py` - Phase 2: Create ordered OG sequences
3. `find_conserved_neighborhoods.py` - Phases 3-4: K-mer counting and merging
4. `evaluate_elsa_orthogroup.py` - Phase 5: Compare ELSA to GT

---

## Open Questions

1. **Orthogroup clustering method**: HDBSCAN vs Leiden vs simple threshold-based?
   - HDBSCAN handles variable density but may split some families
   - Leiden/Louvain on kNN graph might work better

2. **Handling paralogs**: If a genome has 2 copies of an OG, how do we handle the sequence?
   - Option A: Allow OG to appear multiple times in sequence
   - Option B: Mark paralogs specially

3. **Strand handling**: Should "OG1-OG2-OG3" forward equal "OG3-OG2-OG1" reverse?
   - Probably yes for finding conserved neighborhoods

4. **K-mer size**: Fixed k vs variable-length blocks?
   - Could start with k=3, then extend greedily while support stays high
