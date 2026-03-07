# Cross-Species Synteny Benchmark Plan

## Status: Ready to Run

### Dataset
| Species | Genomes | Source |
|---------|---------|--------|
| E. coli | 20 | Existing benchmark data |
| Salmonella enterica | 5 | Downloaded (Typhimurium, Typhi, Enteritidis) |
| Klebsiella pneumoniae | 5 | Downloaded |
| **Total** | **30** | |

### Files Created
- `benchmarks/data/enterobacteriaceae/` - Downloaded genomes, proteins, annotations
- `benchmarks/data/enterobacteriaceae/samples.tsv` - Sample manifest
- `benchmarks/configs/enterobacteriaceae.config.yaml` - ELSA config

### Negative Control (Completed)
E. coli vs B. subtilis embedding similarity:
- Mean cross-species similarity: **0.001** (random)
- Only 0.21% of pairs have >0.8 similarity
- High-similarity pairs are universal housekeeping genes (OG_00000, OG_00001)

**Conclusion:** ELSA won't hallucinate synteny between distant species.

## To Run (When Compute Available)

```bash
# 1. Run ELSA ingest + embed (GPU-intensive)
elsa ingest -c benchmarks/configs/enterobacteriaceae.config.yaml
elsa embed -c benchmarks/configs/enterobacteriaceae.config.yaml

# 2. Run chain analysis
elsa analyze -c benchmarks/configs/enterobacteriaceae.config.yaml \
    -o benchmarks/results/enterobacteriaceae_chain

# 3. Evaluate cross-species synteny
python benchmarks/scripts/evaluate_cross_species.py  # To be written
```

## Expected Results

| Comparison | Expected Synteny | Why |
|------------|------------------|-----|
| E. coli ↔ E. coli | High (~99%) | Same species, established benchmark |
| E. coli ↔ Salmonella | Medium-High | ~40-50% DNA relatedness, same family |
| E. coli ↔ Klebsiella | Medium | More distant, same family |
| Salmonella ↔ Klebsiella | Medium | Similar distance to each other |
| Any ↔ B. subtilis | ~0% | Different phylum (negative control) |

## Metrics to Report

1. **Syntenic block count** per species pair
2. **Gene coverage** - what fraction of genes are in syntenic blocks
3. **Ortholog validation** - do cross-species blocks contain orthologs
4. **Block size distribution** - are cross-species blocks smaller?

## Phylogenetic Context

```
Enterobacteriaceae (family)
├── Escherichia coli
├── Salmonella enterica  (~40-50% DNA relatedness to E. coli)
└── Klebsiella pneumoniae (~40-50% DNA relatedness to E. coli)

Bacillaceae (different phylum - Firmicutes)
└── Bacillus subtilis (negative control)
```

Sources:
- [Enterobacteriaceae phylogenomics](https://pubmed.ncbi.nlm.nih.gov/28658607/)
- [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=562)
