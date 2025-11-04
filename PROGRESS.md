# Project Progress

- [x] 1. Scaffold, CI, PROGRESS.md
- [x] 2. Preprocess (ZCA-LW → PCA → L2)
- [x] 3. Order-invariant shingle vectors
- [x] 4. HNSW index + retrieval
- [ ] 5. Linear metric (ITML) [optional]
- [x] 6. Sinkhorn re-ranking
- [x] 7. Graph + Leiden/HDBSCAN
- [ ] 8. ITQ+MIH path [optional]
- [ ] 9. CLI + config + evaluation

## Notes
- 2025-11-03: Created project scaffold, CLI placeholders, doc stubs, Makefile/tooling, and milestone-aware tests; ran `make fmt`, `make lint`, `make test`.
- 2025-11-03: Implemented Ledoit–Wolf whitening + PCA preprocessor, scripted fitting entrypoint, config-driven CLI integration, and added covariance/unit-norm checks; suite (`make fmt`, `make lint`, `make test`) passes.
- 2025-11-03: Added order-invariant shingle vectors, build-shingles CLI (config-first), regression tests for permutation invariance/CLI artifacts, HNSW indexing with CLI + tests (gracefully skips if hnswlib unavailable), and Sinkhorn rerank command with OT-based similarity + CLI/functional tests.
- 2025-11-03: **Operon micro sidecar** now runs via `elsa analyze --operon` → generates 10,965 blocks, 93k HNSW neighbor pairs, filters to 19,492 Sinkhorn-similarity edges (τ=0.55) grouped into 2,077 connected components (≥2 genomes). Results land in `operon_micro_*` CSVs and SQLite tables (`operon_micro_clusters/blocks/pairs/gene_mappings`). Next steps: (1) dereplicate operon blocks against macro spans, (2) project high-confidence operon clusters into `syntenic_blocks`/`clusters` for browser overlay, (3) surface metrics/filters in Streamlit; legacy `--micro` remains available until explicitly retired.
- 2025-11-03: Added macro-span dereplication to the operon pipeline; `run_operon_pipeline` now removes fully contained pairs before clustering (2547 pairs dropped on the 6-genome dataset) and reports updated counts.
- 2025-11-03: Projected dereplicated operon pairs into the browser DB — 2,031 clusters / 16,945 blocks inserted with `block_type='operon'` and gene-level mappings, enabling immediate overlay alongside macro synteny.
- 2025-11-03: Normalized operon gene→block projections to use browser gene IDs (`accn|…_idx`), wrote 101k mappings, and surfaced cluster-type filters/order in Streamlit (macro/micro/operon). CLI now auto-runs the legacy micro sidecar when `--operon` is set unless the user explicitly passes `--no-micro`.
- 2025-11-03: Merged near-duplicate operon clusters that differed only by ±2-gene offsets across member genomes; cluster count dropped from 2,031 to 956 and projected blocks from 16,945 to 7,383, eliminating redundant overlays for conserved loci.
- 2025-11-04: Added configurable operon merge thresholds (max gap, support ratio) and conservative adjacency merging; operon clusters now default to 287 on the six-genome S. pneumoniae set with rich 10-gene cores preserved without spurious concatenation.
- 2025-11-04: Implemented reciprocal kNN graph construction, Leiden/HDBSCAN clustering utilities, CLI subcommands, and milestone tests; `make fmt`, `make lint`, `make test` all pass.
