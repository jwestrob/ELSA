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


---

2025-09-02 — Micro Pass + UI Integration Progress Note

Summary of what we implemented and fixed since the last note to keep context compact and recoverable.

What’s working now
- Micro-only parameter overrides: Wired `analyze.micro_overrides` so micro pass can be tuned independently of macro. Added working configs:
  - `configs/elsa_full_micro_strict.yaml` (macro preserved + strict micro)
  - Baseline and strict YAMLs preserved in `configs/`.
- Micro dereplication:
  - Pre-cluster filter (DB present): keeps only micro blocks not fully contained in macro spans on both sides.
  - Post-write dedup (CLI): runs after micro sidecar tables are written and before consensus precompute; removes redundant micro blocks and updates micro clusters.
- Micro cluster integration in the browser:
  - Offset micro cluster IDs by `n_macro` (excluding sink) to avoid collisions with macro cluster IDs.
  - Type-aware consensus preview: `_consensus_preview(..., cluster_type='micro')` reads `micro_cluster_consensus` or computes on the fly.
  - Micro card stats: `ClusterAnalyzer` now computes PFAM domain counts and contig/gene scope directly from `micro_gene_*` spans.
  - Micro consensus uses full multi-domain PFAM content per gene (no more first-domain only).
- Diagnostics:
  - Micro pass prints concise “[Micro] …” diagnostics: params used, raw block counts, pre-cluster dedup removed, unique shingles/DF filter results, candidates/edges/mutual edges, and final cluster counts/support.

Cluster Explorer UX
- Multi-locus clinker view (always visible at bottom of Explore Cluster):
  - Adjacent-row edges only; tooltips with cosine and PFAM overlap.
  - Always-on larger rendering + padding; zoom controls (0.6x–3.0x) with dynamic width and redraw.
  - Click-to-center homologs: centers a clicked gene and propagates to the best single neighbor in adjacent rows, chaining across the stack (uses cosine τ; PFAM fallback if no embeddings).
  - Orientation controls: “Flip all to forward”, per-row flip via label, and double-click on a gene to flip that row.

Key fixes
- Template escaping in embedded JS (Streamlit): replaced f-strings with safe placeholders and a normalization pass to convert `{{ }}` into valid JS braces.
- Resolved `innerW` reference after introducing zoom (`baseInnerW*zoom`).
- Correct mapping of display micro IDs to raw IDs for consensus/regions.

Known limitations / follow-ups
- Micro PFAM bar charts use span-based counts; identity/length stats for micro cards are placeholders (macro concepts). Consider adding micro-appropriate summaries (avg genes per block, avg PFAMs per gene).
- Optional: colored edge intensity by cosine / PFAM overlap in the clinker view; per-row “lock” toggle to prevent row motion on centering; “Fit” zoom.
- Lean (no DB) runs skip pre-cluster dedup; consider a minimal temporary DB or alternate path to enable derep even without full browser setup.

Quick usage tips
- Run strict micro-only tuning without changing macro:
  - `elsa analyze --micro -c configs/elsa_full_micro_strict.yaml`
- In the browser:
  - Cluster Explorer → Explore Cluster → bottom “Cluster-wide Clinker Alignment”
  - Zoom +/−, click a gene to center homologs, double-click to flip a row, adjust cosine τ.

2025-09-03 — Clinker UX + Micro Alignments (Current Status)

Problem summary (compact)
- Clinker: centering/zoom drifted content off-screen; edge filtering didn’t respond; protein glyphs were inconsistently scaled across rows; edge coloring/legend unclear.
- Micro: “blocks” were single loci (no target), yet the Block Explorer/UI implied aligned windows; inspecting clusters showed spurious groupings with no high-cosine homologs.

What we changed
- Clinker (Explore Cluster):
  - True pan/zoom using D3 transform; fixed-size canvas; background drag pans all rows; bounded zoom.
  - Centering uses absolute offsets and respects pan/zoom; per-row drag/flip preserved.
  - AA-proportional glyph widths across rows; edges render beneath genes.
  - Cosine τ slider (live filter) + white label; legend with colors: blue (cos-only), orange (PFAM-only), purple (both); usage tips.
  - Embedding status and direct cosine from attached vectors; fixed template/JS errors.

- Micro (alignments + browser parity):
  - Added alignment stage producing two-sided micro_block_pairs and per-gene micro_gene_pair_mappings.
  - Block Viewer now prefers micro mappings/spans to render exact aligned regions for both query and target.
  - Explorer combines macro (syntenic_blocks) with micro pairs; excludes legacy syntenic_blocks ‘micro’ rows (query-only) to avoid confusion.
  - Micro consensus now prefers paired mappings; falls back to legacy only if pairs absent.
  - Fixed contig parsing for micro loci (strip “#start-end” and “:start-end” suffixes); guarded NaNs/None in details panel.

Why earlier results were misleading
- Micro discovery emitted single-locus windows only (no true pair), but the macro-centric viewer inferred “windows” from fields not defined for micro alignments. This showed apparent aligned regions with no target content.

Current status
- Clinker multi-locus view: pan/zoom/centering solid; cosine edges + legend accurate; AA scaling consistent.
- Block Explorer: shows macro + two-sided micro pairs; selecting a micro block opens true query+target with aligned genes.
- Micro consensus: computed from paired mappings when available.

Open items / next steps
- Purity gates for micro: global DF bans, cosine-within-±1 gate, triangle/k-core support to reduce spurious clusters.
- Derep on micro pairs against macro (containment on both sides; favor macro).
- pfam_search for micro in Explorer (join mappings→genes) for feature parity with macro.
- Optional clinker UX: “Fit” zoom, edge hit-layer above genes for better hover.

Operator notes
- If micro pairs don’t appear, confirm DB: SELECT COUNT(*) FROM micro_block_pairs; the browser and analyzer must point to the same genome_browser/genome_browser.db.
- Restart Streamlit after analyze to refresh caches.
