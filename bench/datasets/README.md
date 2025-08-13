# ELSA Phase-2 Benchmark Datasets

This directory contains dataset specifications for benchmarking ELSA Phase-2 improvements against baseline methods.

## Dataset Categories

### 1. Close Divergence (E. coli strains)
**Purpose**: Test syntenic block detection in closely related bacteria
**Expected Behavior**: High synteny conservation, minimal rearrangements

| Genome | NCBI Accession | Size (Mbp) | Description |
|--------|---------------|------------|-------------|
| E. coli K-12 MG1655 | NC_000913.3 | 4.64 | Laboratory reference strain |
| E. coli O157:H7 EDL933 | NC_002655.2 | 5.53 | Pathogenic strain with LEE pathogenicity island |
| E. coli CFT073 | NC_004431.1 | 5.23 | Uropathogenic strain (UPEC) |
| E. coli O104:H4 2011C-3493 | NC_018658.1 | 5.44 | EHEC outbreak strain with aggregative traits |

**Expected Synteny**: ~80-90% conserved backbone with strain-specific insertions

### 2. Moderate Divergence (Corynebacterium species)
**Purpose**: Test cross-species syntenic detection
**Expected Behavior**: Moderate synteny with species-specific adaptations

| Genome | NCBI Accession | Size (Mbp) | Description |
|--------|---------------|------------|-------------|
| C. glutamicum ATCC 13032 | NC_003450.3 | 3.31 | Industrial amino acid producer |
| C. diphtheriae NCTC 13129 | NC_002935.2 | 2.49 | Human pathogen (diphtheria) |
| C. efficiens YS-314 | NC_004369.1 | 3.15 | Thermotolerant industrial strain |

**Expected Synteny**: ~60-70% conserved core with variable accessory regions

### 3. Rearrangement-Rich (Phage/Prophage neighborhoods)
**Purpose**: Test handling of complex rearrangements and mobile genetic elements
**Expected Behavior**: Local synteny around prophage integration sites

| Genome | NCBI Accession | Size (Mbp) | Description |
|--------|---------------|------------|-------------|
| Bacillus subtilis 168 | NC_000964.3 | 4.21 | Multiple prophage regions (SP-β, PBSX, skin) |
| Lactococcus lactis IL1403 | NC_002662.1 | 2.37 | 6 prophage regions, plasmids |
| Streptococcus pyogenes M1 GAS | NC_002737.2 | 1.85 | Multiple prophage and mobile elements |

**Expected Synteny**: Highly variable; synteny broken by prophage insertions

## Benchmark Metrics

### Primary Metrics
1. **Block Boundary F1**: Precision/recall of syntenic block start/end positions
2. **Orientation Accuracy**: Fraction of blocks with correct strand assignments
3. **Continuity**: Fraction of genome covered by collinear blocks
4. **False Discovery Rate**: Empirical FDR vs. target FDR (calibration)

### Secondary Metrics
1. **Memory Efficiency**: Peak memory usage per genome (MB)
2. **Runtime Performance**: Wall-clock time for analysis pipeline
3. **Anchor Density**: Number of anchors per kilobase
4. **Chain Length Distribution**: Statistics on syntenic block sizes

## Baseline Methods

### progressiveMauve
- **Version**: 2.4.0+
- **Parameters**: Default settings, seed length 15
- **Output**: XMFA alignment format
- **Metric extraction**: Convert to block coordinates

### SibeliaZ  
- **Version**: 1.2.0+
- **Parameters**: k=15, minimum block length 500bp
- **Output**: GFF format with syntenic blocks
- **Metric extraction**: Direct coordinate comparison

### SyRI (Optional)
- **Version**: 1.6.0+
- **Requirements**: Whole-genome alignments (minimap2/mummer)
- **Parameters**: Default settings
- **Focus**: Large structural variations

## Dataset Preparation Commands

```bash
# Download and prepare E. coli close divergence set
cd bench/datasets
mkdir -p ecoli_close && cd ecoli_close

# Download genomes (example for one)
wget "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/825/GCF_000005825.2_ASM582v2/GCF_000005825.2_ASM582v2_genomic.fna.gz"
gunzip *.fna.gz

# Download annotations  
wget "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/825/GCF_000005825.2_ASM582v2/GCF_000005825.2_ASM582v2_genomic.gff.gz"
gunzip *.gff.gz

# Repeat for all genomes...
```

## Integration with ELSA

### Test Data Symlinks
The existing `test_data/` genomes should be included in benchmarking:
- `1313.30775` -> Unknown bacterial isolate
- `CAYEVI000000000` -> Clostridium sp.
- `JALJEL000000000` -> Bacillus sp.
- `JBBKAE000000000` -> Streptococcus sp.
- `JBJIAH000000000` -> Enterococcus sp.
- `JBLTKP000000000` -> Lactobacillus sp.

### Benchmark Execution
```bash
# Run ELSA Phase-1 vs Phase-2 comparison
python bench/run_benchmark.py \
  --datasets ecoli_close,corynebacterium,prophage \
  --methods elsa_phase1,elsa_phase2,mauve,sibeliaz \
  --output-dir bench_results/

# Generate comparison report
python bench/generate_report.py \
  --results-dir bench_results/ \
  --output bench_report.html
```

## Expected Outcomes

### Phase-2 Improvements
- **15-25% recall improvement** at fixed 5% FDR on moderate divergence datasets
- **Better calibration** with observed FDR ≤ 1.2× target
- **Reduced false positives** from repetitive elements (MGE masking)
- **Improved orientation detection** (flip-aware chaining)

### Performance Targets
- **Memory**: ≤ 0.6 KB per gene window for sketches
- **Runtime**: Competitive with current implementation (within 1.2×)
- **Scalability**: Linear scaling with genome count for all-vs-all comparisons

## Quality Control

### Dataset Validation
- Verify all genomes download correctly
- Check annotation completeness (>95% CDS coverage)
- Validate PFAM annotation pipeline integration

### Baseline Method QC
- Ensure reproducible baseline method execution
- Cross-validate results between different baseline tools
- Document any baseline method failures or parameter tuning

## Future Extensions

### Additional Dataset Categories
- **Pan-genome diversity**: Multiple strains per species (>10 genomes)
- **Ancient duplications**: Genomes with whole-genome duplication events
- **Plasmid synteny**: Extra-chromosomal element conservation
- **Metagenome fragments**: Incomplete/fragmented assemblies

### Synthetic Benchmarks
- **Simulated rearrangements**: Known ground-truth inversions/translocations
- **MGE insertion/deletion**: Controlled mobile element dynamics
- **Divergence gradients**: Systematic mutation accumulation