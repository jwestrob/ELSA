# Two-Level Windowing Implementation Plan

## Problem Statement

Current ELSA pipeline is aligning entire contigs rather than finding specific syntenic blocks/operons because it lacks proper coordinate-based windowing. We see:
- Length = n_windows (1 bp = 1 window)
- Whole contig alignments instead of gene clusters
- No sub-contig resolution for syntenic block discovery

## Root Cause Analysis

The pipeline currently does:
1. **Gene-level shingling only:** 5 consecutive genes per shingle
2. **No genomic windowing:** Treats entire contigs as single sequences
3. **Result:** Whole-contig matches instead of operons/gene clusters

## Required Two-Level Windowing Architecture

### Level 1: Genomic Coordinate Windows
- **Purpose:** Break contigs into overlapping genomic regions
- **Window size:** 10-20 kb (operon to gene cluster scale)
- **Overlap:** 50% (5-10 kb stride) for boundary spanning
- **Output:** Discrete genomic regions with gene coordinates

### Level 2: Gene Shingles Within Windows
- **Purpose:** Create gene sequence patterns within each genomic window
- **Shingle size:** 5 consecutive genes (current `n: 5`)
- **Stride:** 1 gene step (current `stride: 1`)
- **Output:** Gene sequence signatures for each window

## Implementation Plan

### Phase 1: Understand Current Pipeline Architecture
- [ ] Examine ELSA source code for windowing implementation
- [ ] Identify where genomic windowing should be inserted
- [ ] Map current config parameters to windowing stages

### Phase 2: Identify Configuration Changes
- [ ] Determine missing windowing parameters in `elsa.config.yaml`
- [ ] Research ELSA documentation for proper windowing syntax
- [ ] Create corrected configuration file

### Phase 3: Pipeline Modification
- [ ] Backup current results and configuration
- [ ] Implement two-level windowing configuration
- [ ] Test with subset of data to validate approach

### Phase 4: Re-run Analysis
- [ ] Execute corrected ELSA pipeline
- [ ] Validate that windows << contig length
- [ ] Confirm syntenic blocks are gene cluster scale (5-30 kb)

### Phase 5: Verification
- [ ] Analyze new block size distribution
- [ ] Verify n_windows != length relationship
- [ ] Test genome browser with proper syntenic blocks

## Expected Outcomes

### Before (Current State)
```
Block: 55,749 bp, 55,749 windows
Result: Whole contig alignment
```

### After (Target State)  
```
Block: 12,350 bp, 62 windows (200bp/window)
Result: Specific gene cluster alignment
```

## Configuration Parameters to Add/Modify

```yaml
# Proposed additions to elsa.config.yaml
windowing:
  enable: true
  window_size: 15000      # 15kb genomic windows
  window_stride: 7500     # 50% overlap
  min_genes_per_window: 3 # Require minimum gene density

shingles:
  n: 5                    # Keep current gene shingle size
  stride: 1               # Keep current gene stride  
  # ... other existing parameters
```

## Success Metrics

1. **Window independence:** `n_windows != length` 
2. **Reasonable block sizes:** Most blocks 5-30 kb range
3. **Multiple blocks per contig:** Evidence of sub-contig resolution
4. **Functional relevance:** Blocks correspond to operons/gene clusters

## Risk Assessment

- **Pipeline complexity:** Two-level windowing adds computational overhead
- **Parameter tuning:** May require iteration to find optimal window sizes
- **Data compatibility:** Need to ensure existing analysis tools work with new format
- **Recomputation time:** Full pipeline re-run required

## Next Steps

1. Start with Phase 1: Code examination to understand current architecture
2. Create test configuration with proposed windowing parameters
3. Run on small dataset subset to validate approach
4. Scale to full dataset once parameters are optimized