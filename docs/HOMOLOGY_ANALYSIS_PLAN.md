# Homology Analysis Integration Plan

## Overview

Enhance ELSA's syntenic block analysis with detailed protein homology information using MMseqs2 all-vs-all alignments. This will provide GPT-5 with granular functional relationship data for deeper biological interpretation.

## Architecture

```
Syntenic Block Selection
        ↓
Sequence Extraction (per-contig protein sets)
        ↓  
MMseqs2 All-vs-All Alignment (cross-contig only)
        ↓
Functional Categorization & Pathway Grouping
        ↓
Enhanced DSPy Signature with Homology Data
        ↓
GPT-5 Analysis with Detailed Functional Relationships
```

## Implementation Steps

### Step 1: Planning & Documentation ✅
**Goal**: Create comprehensive plan with testing requirements
**Deliverable**: This planning document
**Testing**: Plan review and validation

### Step 2: Sequence Extraction System
**Goal**: Extract protein sequences from syntenic block regions for MMseqs2 input

**Files to create:**
- `genome_browser/analysis/sequence_extractor.py`

**Requirements:**
- Extract protein FASTA sequences for genes in syntenic blocks
- Separate sequences by contig/genome (query vs target)
- Handle gene coordinate mapping from database
- Validate FASTA format and sequence integrity

**Testing Requirements:**
- [ ] Test with known syntenic block (e.g., block 1574)
- [ ] Verify query proteins != target proteins (no same-contig mixing)
- [ ] Confirm FASTA format validity
- [ ] Check sequence count matches expected gene count
- [ ] Validate all sequences have valid protein characters
- [ ] Test edge cases (missing sequences, malformed coordinates)

**Success Criteria:**
- Extract 2 separate FASTA files (query_proteins.faa, target_proteins.faa)
- No sequence overlap between files
- Valid protein sequences (no nucleotides, stop codons only at end)
- Sequence headers match gene IDs from database

### Step 3: MMseqs2 Integration System
**Goal**: Run cross-contig all-vs-all protein alignment using MMseqs2

**Files to create:**
- `genome_browser/analysis/mmseqs2_runner.py`

**Requirements:**
- Install/check MMseqs2 availability
- Run all-vs-all alignment between query and target protein sets
- Parse MMseqs2 output into structured format
- Handle alignment parameters (sensitivity, e-value thresholds)

**Testing Requirements:**
- [ ] Test MMseqs2 installation and version check
- [ ] Verify no self-hits (protein vs itself)
- [ ] Confirm only cross-contig alignments (query vs target, not query vs query)
- [ ] Validate alignment results have expected columns
- [ ] Check e-value and identity score ranges
- [ ] Test with different parameter settings
- [ ] Handle empty/no-hit results gracefully

**Success Criteria:**
- MMseqs2 runs without errors
- Output contains only cross-contig protein pairs
- Alignment results include: query_id, target_id, identity, coverage, e-value
- No contamination from same-contig comparisons
- Reasonable number of significant hits (not all/none)

### Step 4: Functional Categorization System
**Goal**: Process MMseqs2 results into functional relationship categories

**Files to create:**
- `genome_browser/analysis/homology_processor.py`

**Requirements:**
- Parse MMseqs2 output into structured homology data
- Categorize relationships (ortholog, paralog, no_homology)
- Group proteins by functional units/pathways
- Map alignments to PFAM domain conservation
- Generate pathway-level conservation metrics

**Testing Requirements:**
- [ ] Test relationship categorization accuracy
- [ ] Verify ortholog detection (high identity + coverage)
- [ ] Confirm no_homology filtering (low scores)
- [ ] Test functional grouping logic
- [ ] Validate PFAM domain mapping
- [ ] Check pathway conservation calculations
- [ ] Test with various alignment quality ranges

**Success Criteria:**
- Accurate ortholog identification (>70% identity, >80% coverage)
- Proper functional grouping by pathway/domain families
- Conservation metrics match biological expectations
- Clear distinction between significant and background hits
- Structured output ready for GPT-5 consumption

### Step 5: Enhanced DSPy Integration
**Goal**: Integrate homology data into GPT-5 analysis pipeline

**Files to modify:**
- `genome_browser/analysis/gpt5_analyzer.py`
- `genome_browser/analysis/data_collector.py`

**Requirements:**
- New DSPy signature with homology inputs
- Data collector integration with homology pipeline
- Enhanced GPT-5 prompts for functional analysis
- Structured homology output formatting

**Testing Requirements:**
- [ ] Test new DSPy signature with homology data
- [ ] Verify data collector integration
- [ ] Confirm GPT-5 receives properly formatted homology info
- [ ] Test enhanced analysis quality vs baseline
- [ ] Validate output includes homology insights
- [ ] Check performance (analysis time < 60 seconds)

**Success Criteria:**
- GPT-5 analysis includes detailed functional relationships
- Homology data enhances biological interpretation quality
- Analysis maintains reasonable performance
- Output clearly distinguishes homology-informed insights
- No degradation in baseline functional analysis

## Data Flow Architecture

### Input Data
```python
syntenic_block = {
    "block_id": "1574",
    "query_locus": "CAYEVI000000000.1_region_1",
    "target_locus": "1313.30775.1_region_2", 
    "genes": [gene_objects...]
}
```

### Sequence Extraction Output
```python
extracted_sequences = {
    "query_proteins": "query_proteins.faa",  # 10 proteins
    "target_proteins": "target_proteins.faa",  # 12 proteins
    "gene_mapping": {
        "query_gene_1": "protein_seq_1",
        "target_gene_1": "protein_seq_1"
    }
}
```

### MMseqs2 Output
```python
alignments = [
    {
        "query": "query_gene_1",
        "target": "target_gene_3", 
        "identity": 0.94,
        "coverage": 0.88,
        "evalue": 1e-120,
        "bitscore": 245.2
    }
]
```

### Functional Categorization Output
```python
homology_analysis = {
    "ortholog_pairs": [
        {
            "query": "query_gene_1",
            "target": "target_gene_3",
            "relationship": "strong_ortholog",
            "functional_conservation": "catalytic_core_identical",
            "pathway_role": "dTDP_glucose_synthesis"
        }
    ],
    "pathway_conservation": {
        "dTDP_rhamnose_pathway": "complete",
        "conservation_level": "very_high"
    }
}
```

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Mock data for reproducible testing
- Edge case handling validation
- Error condition testing

### Integration Tests  
- End-to-end pipeline testing
- Real syntenic block data validation
- Performance benchmarking
- Data consistency checks

### Validation Tests
- Compare with known biological relationships
- Cross-reference with literature/databases
- Manual spot-checking of results
- False positive/negative analysis

## Success Metrics

1. **Accuracy**: Homology relationships match biological expectations
2. **Performance**: Analysis completes in <60 seconds for typical blocks
3. **Coverage**: Successfully processes >90% of syntenic blocks
4. **Quality**: GPT-5 analysis shows clear improvement with homology data
5. **Reliability**: Consistent results across multiple runs

## Risk Mitigation

- **MMseqs2 dependency**: Check installation, provide clear setup instructions
- **Sequence quality**: Validate input sequences before alignment
- **Alignment sensitivity**: Test multiple parameter sets for optimal results
- **Data integration**: Robust error handling for missing/malformed data
- **Performance**: Optimize for reasonable analysis times

## Future Enhancements

- Multiple alignment algorithms (BLAST, DIAMOND)
- Structural homology prediction
- Phylogenetic analysis integration
- Interactive homology visualization
- Batch processing for multiple blocks

---

**Implementation Order**: Steps 2 → 3 → 4 → 5, with rigorous testing at each stage.
**Testing Philosophy**: No shortcuts, validate every assumption, test edge cases.
**Goal**: Provide GPT-5 with rich, accurate homology data for superior biological analysis.