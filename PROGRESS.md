# ELSA Implementation Progress

## Overview
Implementing ELSA (Embedding Locus Shingle Alignment) - a bioinformatics tool for syntenic block discovery using protein language model embeddings.

**Target Platform:** M4 Max MacBook Pro, 48GB RAM, GPU-enabled with CPU fallback

## Implementation Sequence (from spec section 10)

### Phase 1: Core Infrastructure ✅
- [x] 1. Skeleton: `elsa/` package, `cli.py`, `api/service.py`, `params.py`
- [x] 2. IO: `manifest.py`, `ingest.py`, `embeddings.py` readers

### Phase 2: Embedding Pipeline ✅  
- [x] 3. Embedding: ESM2/ProtT5 loaders, GPU batching, mean-pooling
- [x] 4. Projection: PCA fit/transform, `genes.parquet` output
- [x] 5. Shingling: window generation with positional encoding

### Phase 3: Indexing 🔄
- [ ] 6. Discrete: codebook training, n-gram hashing, MinHash LSH
- [ ] 7. Continuous: SRP hyperplanes, signatures, Hamming search

### Phase 4: Search & Chain 🔄
- [ ] 8. Search: anchor scoring, chaining DP, optional DTW

### Phase 5: Interfaces 🔄
- [x] 9. CLI: wire all commands with progress bars
- [ ] 10. Service: FastAPI endpoints for `/find_like_block`

### Phase 6: Quality & Docs 📊
- [ ] 11. QC: plotting utilities, collision analysis
- [ ] 12. Docs: overview, CLI, file formats, API examples

## Current Status: Core Pipeline COMPLETE! 🎉
- ✅ **Complete embedding pipeline**: nucleotide FASTA → proteins → embeddings → PCA → windows
- ✅ **Directory-based interface**: `elsa embed data/` auto-discovers and processes all genomes
- ✅ **GPU-optimized**: ESM2/ProtT5 with M4 Max MPS acceleration  
- ✅ **Production-ready**: Manifest tracking, checkpointing, error handling
- 🔄 **Next: Indexing systems** for syntenic block discovery

## GPU Optimization Notes
- **PLM Inference**: ProtT5/ESM2 models benefit significantly from GPU
- **Batch Processing**: Large protein batches on GPU, memory-efficient streaming
- **Fallback Strategy**: Auto-detect MPS/CPU, graceful degradation
- **Memory Management**: Monitor 48GB envelope, use memory mapping

## Questions & Decisions
1. **PLM Model Priority**: Start with ProtT5 or ESM2? (spec mentions both)
2. **Dependency Management**: Use conda/pip for bio dependencies?
3. **Testing Data**: Need sample FASTA files for development?
4. **GPU Memory**: How to optimally batch on M4 Max unified memory?

## Next Steps
1. Update CLAUDE.md for GPU capabilities
2. Set up initial package structure
3. Implement configuration system
4. Begin with PLM embedding pipeline (GPU-first)