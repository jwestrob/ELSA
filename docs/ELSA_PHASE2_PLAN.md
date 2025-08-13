# ELSA Phase-2 Implementation Plan

## Overview

Phase-2 upgrades ELSA's syntenic block detection with weighted sketches, multi-scale anchoring, flip-aware chaining, statistical calibration, and dense retrieval. All improvements are feature-flagged to preserve current behavior by default.

## Core Improvements

### 1. Weighted Sketching + IDF + MGE Masking
**Problem**: Current MinHash treats all genes equally, causing repetitive elements to dominate similarity.
**Solution**: Weight codewords by IDF; mask MGE-associated PFAM domains; use b-bit compression.

### 2. Multi-Scale Windowing
**Problem**: Fixed 5-gene windows miss both fine-grained and coarse-grained synteny patterns.
**Solution**: Macro windows (8-20 genes) for candidate filtering → Micro windows (3-5 genes) for refinement.

### 3. Flip-Aware Affine-Gap Chaining
**Problem**: Current chaining ignores strand flips and local rearrangements.
**Solution**: 2-state DP (forward/reverse) with explicit flip penalties + optional DTW refinement.

### 4. Statistical Calibration
**Problem**: No principled threshold for syntenic significance.
**Solution**: Null models via permutation → per-chain p-values → FDR control.

### 5. Dense Retrieval (HNSW)
**Problem**: SRP prefiltering may miss high-similarity candidates.
**Solution**: HNSW index on shingle centroids; SRP remains as optional prefilter.

## Configuration Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `elsa.phase2.enable` | `false` | Master switch for all Phase-2 features |
| `elsa.phase2.weighted_sketch` | `false` | Use weighted MinHash with IDF/MGE masking |
| `elsa.phase2.multiscale` | `false` | Enable macro→micro windowing |
| `elsa.phase2.flip_dp` | `false` | Use flip-aware affine-gap chaining |
| `elsa.phase2.calibration` | `false` | Enable null models and FDR control |
| `elsa.phase2.hnsw` | `false` | Use HNSW dense retrieval |

### Weighted Sketching Parameters
```yaml
elsa:
  sketch:
    type: "minhash"  # or "weighted_minhash"
    bits: 64        # or 1,2 for compression
    size: 96        # sketch size
    idf:
      max: 10.0     # clamp IDF values
  mge_mask:
    path: null      # YAML list of PFAM accessions to mask
```

### Multi-Scale Window Parameters
```yaml
elsa:
  window:
    micro:
      size: 5
      stride: 1
    macro:
      size: 12
      stride: 3
    adaptive:
      enable: false  # future: CUSUM change-point detection
```

### Chaining Parameters
```yaml
elsa:
  chain:
    alpha: 0.1     # position deviation penalty
    beta: 1.0      # gap penalty
    gamma: 2.0     # strand flip penalty
    dtw:
      enable: false
      band: 2      # DTW band width in genes
```

### Calibration Parameters
```yaml
elsa:
  calib:
    null:
      iters: 100   # null model iterations
    target_fdr: 0.05 # FDR threshold
```

### HNSW Parameters
```yaml
elsa:
  hnsw:
    M: 16
    efConstruction: 200
    efSearch: 50
  retrieval:
    dense: "srp"    # or "hnsw"
```

## Data Flow Diagram

```
Input Genomes
      ↓
  Gene Calling & PLM Embedding
      ↓
  Multi-Scale Windowing
      ↓ (macro)        ↓ (micro)
  Weighted Sketching   Dense Embedding
      ↓                    ↓
  IDF + MGE Masking    HNSW Index
      ↓                    ↓
  Candidate Retrieval ←────┘
      ↓
  Flip-Aware Chaining
      ↓
  Calibration & FDR Filtering
      ↓
  Syntenic Blocks
```

## Flip-Aware Chaining State Machine

```
State: F (Forward)    State: R (Reverse)
     ↓ F→F                 ↓ R→R
   score(F)              score(R)
     ↓                     ↓
     └─────── F↔R ─────────┘
           (penalty γ)

Transitions:
- F→F: same strand, gap penalty β
- R→R: same strand, gap penalty β  
- F→R, R→F: strand flip, penalty γ + β
```

## Implementation Phases

### Phase 1: Weighted Sketching (Priority 1)
**Files:**
- `elsa_index/sketch/weighted_minhash.py`
- `elsa_index/sketch/idf_stats.py`
- `elsa_index/sketch/mge_mask.py`

**Tests:**
- Estimator accuracy vs exact Jaccard
- IDF deterministic propagation
- MGE masking effect on repetitive domains

### Phase 2: Multi-Scale Windowing (Priority 2)
**Files:**
- `elsa/windowing/multiscale.py`

**Tests:**
- Macro prefilter recalls micro candidates
- Window-to-gene mapping correctness

### Phase 3: Flip-Aware Chaining (Priority 3)
**Files:**
- `elsa/chaining/flip_affine.py`
- `elsa/chaining/dtw_refine.py`

**Tests:**
- Single inversion recovery
- Gap penalty calibration
- DTW refinement on small permutations

### Phase 4: Calibration (Priority 4)
**Files:**
- `elsa/eval/null_models.py`
- `elsa/eval/calibration.py`

**Tests:**
- FDR control on permuted genomes
- P-value calibration curves

### Phase 5: HNSW Dense Retrieval (Priority 5)
**Files:**
- `elsa_index/dense/hnsw.py`

**Tests:**
- Recall vs SRP at matched latency
- Memory usage within targets

### Phase 6: UI Enhancements (Priority 6)
**Files:**
- `genome_browser/components/dotplot.py`
- Updates to `genome_browser/app.py`

**Features:**
- Dotplot/synteny view with brush-to-zoom
- FDR slider and role-aware filters
- Per-chain statistics display

## Test Matrix

| Component | Unit Test | Integration Test | Performance Test |
|-----------|-----------|------------------|------------------|
| Weighted MinHash | Estimator accuracy | End-to-end A/B vs current | Memory ≤0.6KB/window |
| Multi-scale | Window mapping | Recall superset check | Latency competitive |
| Flip chaining | Inversion recovery | Known syntenic loci | Gap penalty sensitivity |
| Calibration | Permutation nulls | FDR control validation | P-value computation time |
| HNSW | Index construction | Recall vs SRP | Query latency p95 |

## Integration Dataset

**Small Public Genome Set** (accessions to be added to `bench/datasets/README.md`):
- **Close divergence**: 3-4 E. coli strains 
- **Moderate divergence**: 2-3 Corynebacterium species
- **Rearrangement-rich**: 2-3 genomes with known prophage insertions

## Acceptance Criteria

1. **Recall Improvement**: ≥15% recall gain at fixed 5% FDR vs Phase-1
2. **FDR Control**: Observed FDR ≤1.2× target on permuted nulls
3. **Memory Efficiency**: Sketch memory ≤0.6KB/window at m=96, b=2
4. **UI Functionality**: Dotplot, FDR slider, per-chain stats with synchronized selections
5. **Backward Compatibility**: No behavior change when `elsa.phase2.enable=false`

## Risk Mitigation

- **Estimator Drift**: Keep 64-bit fallback; cross-check against datasketch
- **HNSW Dependencies**: Make pluggable; seamless SRP fallback
- **False Positives**: IDF + MGE masking + masked-fraction thresholds
- **Performance Regression**: A/B test all changes; feature flags for rollback

## Migration Strategy

1. **Feature Flags**: All new behavior gated by granular flags
2. **A/B Testing**: Compare Phase-2 vs current on same datasets
3. **Gradual Rollout**: Enable features incrementally after validation
4. **Compatibility**: Preserve current API; add new endpoints for Phase-2 features

## Dependencies

- **hnswlib** (MIT): Dense ANN indexing
- **datasketch** (test only): MinHash reference implementation
- **scikit-learn**: Existing PCA dependency

## Success Metrics

- Improved recall/precision on moderate-divergence bacteria
- Controlled FDR with statistical significance
- Enhanced user experience with dotplot and filtering
- Maintainable codebase with comprehensive test coverage