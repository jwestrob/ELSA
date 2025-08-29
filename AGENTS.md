# Agent TODOs: Micro-Pass (Embedding‑First) + Dereplication

This doc captures the current plan for implementing a robust, PFAM‑agnostic micro pass that runs alongside the macro pipeline, and how we dereplicate micro hits against larger (macro) syntenic blocks. It replaces any prior notes/experiments and is the single source of truth for the next implementation round.

## Objectives

- Keep the macro pipeline unchanged (5‑gene windows, PFAM‑agnostic), including its clustering and outputs.
- Add a micro pass that discovers 2–3 gene cassettes using embedding signals only.
- Ensure dereplication: micro blocks entirely contained within macro block regions on both sides are removed from micro results (or not projected into main tables) to avoid duplication.
- Keep micro results sidecar by default (TSVs + dedicated DB tables) to avoid inflating macro cluster counts; optional overlay in the browser.

## Micro‑Pass Clustering (Embedding‑First, Gene‑Level)

- Tokens (per gene):
  - Prefer embedding codewords from `elsa_index/shingles/codewords.parquet` → token `CW<id>`.
  - Do not rely on PFAM for membership; PFAM can be used later only for labeling.

- Shingling (per contig):
  - Build short loci (“micro blocks”) using gene k‑grams: k ∈ {2,3}.
  - Allow small gap tolerance: `max_gap = 1` (skip‑1 variants) so PFAM‑less or missing genes do not break cassettes.
  - Represent each micro block as a small set of hashed gene‑kgram shingles (uint64).

- IDF + Candidates:
  - Compute document frequency (DF) over gene‑kgram shingles across all micro blocks.
  - Build an inverted index: shingle → [block ids], drop high‑DF shingles aggressively (`df_max ≈ 50`).
  - Enumerate candidate neighbors via postings (same shingle), dedupe, and compute an IDF‑weighted Jaccard between shingle sets.

- Graph + Clustering:
  - Keep an edge (A,B) when weighted Jaccard ≥ `jaccard_tau` (≈ 0.65–0.75) and passes mutual‑top‑k.
  - Form clusters as connected components (micro is small; community detection optional).
  - Filter clusters by genome support: require ≥ `min_genome_support` genomes (e.g., 3–4) across member blocks.

- Outputs (sidecar):
  - CSVs under `syntenic_analysis/micro_gene/`:
    - `micro_gene_blocks.csv`: `block_id, cluster_id, genome_id, contig_id, start_index, end_index`
    - `micro_gene_clusters.csv`: `cluster_id, size, genomes`
  - DB tables (do not touch `syntenic_blocks`/`clusters`):
    - `micro_gene_blocks`, `micro_gene_clusters`

- Consensus (future enhancement):
  - For UI labeling only: group genes across blocks by cosine ≥ 0.80 within a small positional band (±1), label groups by PFAM majority (if present) else `~E:Cnn`. This does not affect clustering.

## Dereplication Policy (Micro vs Macro)

- Goal: Avoid keeping micro blocks that are redundant with existing macro blocks.

- Method (when projecting micro into the main UI or if we need to filter sidecar results):
  1) For each micro block, compute its per‑side genomic span using `gene_block_mappings` → `genes.(start_pos,end_pos)`:
     - Query span: `[min(start_pos(query genes)), max(end_pos(query genes))]` on `query_contig_id`.
     - Target span: analogous on `target_contig_id`.
  2) For macro blocks (non‑micro), precompute per‑side spans similarly.
  3) Delete the micro block if both its query span is fully contained within some macro query span (same genome+contig) AND its target span is fully contained within some macro target span (same genome+contig).
  4) After deletion, update any micro cluster sizes or drop empty micro clusters, if clusters are materialized in DB.

- Default stance: Keep micro sidecar only (no projection to `syntenic_blocks`/`clusters`), so dereplication is only applied when projecting micro blocks into the main UI. The SQL template for containment exists and should be applied if/when projection is enabled.

## Analyze Integration

- Run macro pipeline as usual.
- If `--micro` is set:
  - Run the micro pass (as above) to produce sidecar CSVs + `micro_gene_*` tables.
  - Do NOT inject micro results into `syntenic_blocks`/`clusters` by default.
  - Keep dereplication routine ready for the optional projection path.

## Parameters (initial)

- `k = 2`, `max_gap = 1`
- `jaccard_tau ≈ 0.65–0.75`, `mutual_k = 3`
- `df_max ≈ 50` (gene‑shingles)
- `min_genome_support = 3` (or 4) per micro cluster

Adjust after first run on the 6‑genome set.

## Validation & Safety

- Determinism: seed all random choices (when added); sorting where needed; stable hashing.
- Sanity checks: counts of genes with tokens, #blocks pre/post DF filter, #edges kept, #clusters, genome support distribution.
- Performance: postings‑based neighbors only; no all‑pairs; skip heavy DTW for micro.

## Browser UX (now)

- Preserve the Micro‑Synteny tab to explore sidecar micro results.
- Optional: add a simple listing for `micro_gene_clusters` and their blocks; provide overlay in Genome Viewer later.

## Out of Scope (for now)

- Injecting micro blocks into `syntenic_blocks`/`clusters` by default.
- Per‑gene embedding DTW for micro edges (use weighted Jaccard first; add DTW later if needed).

