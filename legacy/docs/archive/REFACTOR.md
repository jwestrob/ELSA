# REFACTOR.md - Cassette Mode Implementation Plan

## Current Architecture Analysis

### Core Components Located:

#### 1. **Anchor Generation & Window Matching**
- **Location**: `elsa/search.py` - `SimilaritySearcher` class
- **Current Logic**: 
  - `search_discrete()`: LSH queries with no similarity threshold
  - `search_continuous()`: SRP queries using Hamming distance -> similarity conversion
  - Returns all matches without filtering

#### 2. **Chaining Implementation**  
- **Location**: `elsa/search.py` - `CollinearChainer.chain_matches()`
- **Current Issues**:
  - **Simple grouping by target locus** (no real chaining)
  - **No positional constraints** or gap penalties
  - **No local gain calculation** 
  - **TODO comment**: "Implement proper dynamic programming chaining"
- **Location**: `elsa/analysis.py` - `AllVsAllComparator.compare_loci()`
  - More sophisticated logic with positional filtering
  - Uses `_filter_positional_conservation()` and `_find_consecutive_matches()`

#### 3. **Configuration System**
- **Location**: `elsa/params.py` (need to examine)
- **Config Loading**: `elsa/cli.py` (need to examine)  
- **Current Config**: `elsa.config.yaml` - has chain parameters but they're not being used properly

#### 4. **Window Data Structure**
- **Storage**: Parquet files with embedding columns `emb_000` - `emb_255`
- **Format**: Each window has `window_id`, `sample_id`, `locus_id`, `window_idx`, + embeddings

### Current Problems Identified:

1. **No anchor filtering** - accepts all LSH/SRP matches
2. **Naive chaining** - groups by locus, no positional logic
3. **No gap/position penalties** applied
4. **Config parameters ignored** in search pipeline
5. **Two different implementations** (search.py vs analysis.py)

## Implementation Strategy

Based on GPT-5's plan and patch corrections, focus on **core improvements first**:

### Phase 1: Core Chain Tightening (Immediate Impact)

#### A. Update Window Configuration
- **File**: `elsa.config.yaml`  
- **Change**: `window.micro.size: 2` (currently 3)
- **Impact**: Smaller base window unit

#### B. Implement Two-Key Anchor Gate (search.py)
- **Function**: `SimilaritySearcher.search_discrete()` and `search_continuous()`
- **Add Filters**:
  1. **Cosine threshold**: Only matches with `cosine ≥ 0.91`
  2. **Jaccard threshold**: Only matches with `jaccard ≥ 0.30`  
  3. **Reciprocal top-k**: Mutual best matches only
  4. **Collision blacklist**: Filter high-frequency windows

#### C. Improve Chaining Logic (search.py or analysis.py)
- **Target**: `CollinearChainer.chain_matches()` or `AllVsAllComparator`
- **Add Local Gain Calculation**:
  ```
  ΔS = cosine + λ·jaccard - α·|Δpos| - β·gap_genes
  Accept step only if ΔS ≥ τ (delta_min)
  ```
- **Add Constraints**:
  - `|Δpos| ≤ 1` (tight positional band)
  - `max_gap_genes ≤ 1`
  - **Density floor**: Last 10 genes must have `anchors/gene ≥ 0.3`

#### D. Disable/Constrain DTW
- **Config**: `dtw.enable: false` or `dtw.band: 1`

### Phase 2: Configuration Integration

#### E. Add Cassette Mode Config
- **File**: `elsa.config.yaml`
- **Add Section**:
```yaml
# Cassette mode configuration
cassette_mode:
  enable: false
  anchors:
    cosine_min: 0.91
    jaccard_min: 0.30
    reciprocal_topk: 2
    blacklist_top_pct: 1.0
  chain:
    delta_min: 0.02
    density_window_genes: 10
    density_min_anchors_per_gene: 0.3
    pos_band_genes: 1
    max_gap_genes: 1
```

### Files to Modify

#### Primary Targets:
1. **`elsa/search.py`** - Add anchor filtering and improve chaining
2. **`elsa.config.yaml`** - Add cassette mode config + reduce window size
3. **`elsa/params.py`** - Add config validation for new parameters

#### Secondary (if needed):
4. **`elsa/cli.py`** - Add CLI flags for cassette mode
5. **`elsa/analysis.py`** - Apply same logic to all-vs-all comparisons

## Implementation Priority

### FIRST: Window Size + Basic Filtering
1. Update `elsa.config.yaml`: `window.micro.size: 2`  
2. Add basic anchor filtering to `SimilaritySearcher`
3. Test with rebuild to see immediate impact

### SECOND: Chaining Improvements  
4. Implement local gain logic in `CollinearChainer`
5. Add position/gap constraints
6. Add density floor termination

### THIRD: Configuration System
7. Add cassette_mode config section
8. Wire config into search pipeline
9. Add CLI integration

## Questions Resolved

1. **Architecture**: Found two chaining implementations - will start with `search.py` 
2. **File modifications**: Prefer modifying existing files (`search.py`, `analysis.py`)
3. **Config integration**: Main config file, add cassette section
4. **Approach**: Core first (B1 + A2), then configuration
5. **RANSAC vs Chain**: Start with chain tightening, add RANSAC if needed
6. **Window size**: Update to 2 genes first
7. **No length caps**: Per patch - use evidence-based termination only

## Next Steps

1. Update window size configuration
2. Implement anchor filtering in SimilaritySearcher  
3. Improve chaining with local gain + constraints
4. Test and iterate

## Code Locations Reference

- **Anchor generation**: `elsa/search.py:174-214`
- **Chaining**: `elsa/search.py:222-258` (simple) + `elsa/analysis.py:161-236` (sophisticated)
- **Config**: `elsa.config.yaml` + `elsa/params.py`
- **CLI**: `elsa/cli.py`
- **Window data**: Parquet with `emb_000`-`emb_255` columns