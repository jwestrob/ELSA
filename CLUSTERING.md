# ELSA Syntenic Block Clustering — Technical Design (For Model Review)

This document describes the clustering subsystem that assigns syntenic blocks to cassette families using mutual-Jaccard over SRP-derived shingles. Hybrid augmentation and mutual-top-k gating exist as optional scaffolding but are not part of the default path. It is intentionally dense and implementation-proximal.

---

## 1. Objectives and Scope

- Group collinear syntenic blocks into “cassette” families across genomes.
- Leverage continuous embeddings (SRP) to obtain order-aware, strand-insensitive token sequences.
- Use a sparse similarity graph with conservative pruning; detect communities via modularity.
- Preserve determinism and re-runnability across machines and seeds.
- Optional (off by default): augment edges for long, high-identity operons where XOR shingling is brittle.

Non-goals: Inferring evolutionary history; gene-by-gene orthology; plasmid-level topology.

---

## 2. Data Model

Block b has:
- `query_windows`, `target_windows`: ordered IDs of windows in the collinear chain.
- `matches`: list of paired windows (query_window_id, target_window_id) used for robustness gating.
- `alignment_length` (|matches|), `identity` (mean match similarity), 

Windows W have embeddings E ∈ ℝ^D (post-PLM projection). Per-sample, windows are indexed in Parquet.

---

## 3. Robustness Gate (R)

Filter blocks before tokenization (strict defaults):
- anchors: n ≥ `min_anchors`
- span: both query and target spans (in window indices) ≥ `min_span_genes`
- collinearity: MAD of (q_idx - t_idx) ≤ `v_mad_max_genes`
- cassette mode: if 2 ≤ n ≤ `cassette_max_len` and perfect collinearity, admit.

Fail ⇒ sink (cluster 0). Pass ⇒ robust set 𝓡.

---

## 4. SRP Tokenization (T)

Given matrix X ∈ ℝ^(L×D) for L windows in strand order (reverse if strand = −1):
- Sample R ∈ ℝ^(D×B) with fixed seed; normalize columns; compute bits B=256: s = (X·R ≥ 0).
- Partition into `n_bands` bands of `band_bits` each (e.g., 32×8). For each window and band, hash the band’s bits to a 64-bit token via blake2b.
- Per window i ⇒ list of band tokens τ_i = [t_i0, …, t_i(n_bands−1)].

---

## 5. Shingling (S)

Construct order-aware k-grams over per-window tokens to form a set S_b of shingle IDs (64-bit).

Methods:
- xor: per window, collapse band tokens by XOR to one 64-bit token; shingle size k=`shingle_k` across the sequence. Fast, brittle across near-boundary SRP flips.
- subset: per window, select a deterministic rotating subset of bands (bands_per_window, band_stride), hash their concatenation; then k-grams. More stable than XOR, still brittle in some corpora.
- bandset (auxiliary only): union of all band tokens across windows (order-agnostic). Used for hybrid augmentation (Section 9), not as the primary graph signal.

Strand-insensitive tokens and skip-gram shingling
Before SRP, we null the strand-sign dimension in window embeddings so per-window tokens are the same on forward/reverse loci. Order-awareness is preserved via k-grams.

Skip-gram shingling: Enable k=3 skip-grams using offsets like (0,2,5). This increases robustness to small insertions/deletions while preserving order semantics. If `shingle_pattern` is not set, contiguous k-grams are used.

Parameters: `shingle_method ∈ {xor, subset, bandset, icws}`, `icws_r`, `icws_bbit`, `shingle_pattern`.
Determinism: A global seed (`srp_seed`) deterministically derives per-window/per-sample sub-seeds.

Migration: keep `shingle_method: xor` to reproduce legacy behavior. To try ICWS while preserving defaults elsewhere, set `shingle_method: icws` and optionally `shingle_pattern: "0,2,5"`.

---

## 6. DF Filtering + IDF (F)

- Build inverted index postings for shingles: P[s] = {b | s ∈ S_b}.
- Compute DF(s) = |P[s]|.
- Filter S_b ← {s ∈ S_b | DF(s) ≤ `df_max`}.
- Rebuild postings P′ from filtered sets.
- IDF(s) = log(1 + N/DF(s)) with N = |{b | S_b ≠ ∅}|.

Optionally ban top-percentile DF (max_df_percentile), then rebuild.

For diagnostics only: we may compute order-agnostic bandset overlap to confirm content similarity; the default workflow does not add bandset-derived edges.

---

## 7. Candidate Generation (C)

For each block b with filtered S_b:
- Accumulate candidate counts via postings: c[x] = |S_b ∩ S_x| using P′.
- Keep x with c[x] ≥ `min_shared_shingles`.
- Size prefilter: let s_b = |S_b|, s_x = |S_x|; and alignment lengths n_b, n_x. Accept if `size_ratio_min` ≤ s_b/s_x ≤ `size_ratio_max` and same for n_b/n_x.
- Keep top-`max_candidates_per_block` by c[x].

No hybrid augmentation in the default path.

Former hybrid augmentation (qualified b): add up to `bandset_topk_candidates` candidates x by shared band tokens ≥ `min_shared_band_tokens`, with same size prefilters. Qualification: n_b ≥ `bandset_min_len` and identity ≥ `bandset_min_identity`.

Qualification is symmetric per-pair in scoring; augmentation is per-node in candidate assembly.

---

## 8. Similarity and Edge Creation (E)

For each b and its bounded candidate list C_b:
- Compute weighted Jaccard over shingles:
  J_w(b,x) = Σ_{s ∈ S_b∩S_x} IDF(s) / Σ_{s ∈ S_b∪S_x} IDF(s).
- If J_w ≥ `jaccard_tau`, require informative overlap:
  - low-DF anchors: |{s ∈ S_b∩S_x | DF(s) ≤ low_df_threshold}| ≥ `min_low_df_anchors`
  - mean IDF over intersection ≥ `idf_mean_min`.
- If accepted: add undirected edge {b,x} with weight = J_w.

Optional local cap (off by default): if `enable_mutual_topk_filter` true, restrict to top-`mutual_k` candidates per b before materializing edges (approximate mutual-k gating). Full symmetric mutual-k can be added if needed.

---

## 9. Hybrid Bandset Augmentation (H) — optional (off by default)

For long, highly conserved operons, XOR/subset shingles can 0-out despite high cosine and near-identity. The bandset channel recovers a weak but robust signal.

For qualified pairs (b,x):
- B = bandset(b), X = bandset(x) after DF filtering with `bandset_df_max`.
- Weighted bandset Jaccard:
  J_bw(b,x) = Σ_{t ∈ B∩X} IDF_b(t) / Σ_{t ∈ B∪X} IDF_b(t).
- If no shingle-edge was added and J_bw ≥ `bandset_tau` ⇒ add edge with weight = J_bw.

This yields recall for conserved loci (e.g., ribosomal operons) when needed, but the default profile relies on the main shingle channel for compactness.

---

## 10. Graph Pruning and Communities (G)

- Degree cap: keep top-`degree_cap` incident edges per node by weight.
- Optional k-core/triangle filters are available but disabled by default to keep cassettes intact.
- Community detection: greedy modularity (NetworkX), weight-aware, deterministic under fixed inputs. Fallback: connected components.
- Cluster assignment: sort components by (−size, min block_id) for deterministic IDs: 1..K, with 0 reserved for sink (non-robust and discarded units).

---

## 11. Determinism and Reproducibility (D)

- SRP seed, PLM projection, and config are recorded. With identical inputs, cluster IDs are stable.
- Assignment only depends on filtered sets, IDFs, and graph construction; no RNG beyond SRP.

Suggested fingerprint (print-once):
- hashes: SRP(seed,bits,bands), PCA hash, dataset (N_windows,N_blocks), DF spectra checksum, config JSON.

---

## 12. Complexity and Scaling (X)

Let R = |𝓡| robust blocks, S = average |S_b|.
- Tokenization: O(Σ L_b·D + R·B) (matrix multiply dominates once per block; D fixed).
- Postings build: O(Σ S_b).
- Candidate enumeration: O(Σ S_b·avg_df) but bounded by min_shared filters and caps.
- Scoring: O(Σ |C_b|) set ops over ints, dominated by union/intersection cost (fast in Python sets).
- Community detection: near-linear in edges for greedy modularity on these sparse graphs.

Parallelization: embarrassingly parallel scoring per b once postings/candidates are built (process pool). Current implementation runs single-process by default.

---

## 13. Parameters (P) — Key

Robustness:
- min_anchors, min_span_genes, v_mad_max_genes, cassette_max_len

SRP/shingling:
- srp_bits=256, srp_bands=32, srp_band_bits=8, srp_seed
- shingle_method ∈ {xor, subset}, shingle_k
- subset: bands_per_window, band_stride

Filtering:
- df_max (shingles), max_df_percentile (optional)

Similarity and edges:
- jaccard_tau, use_weighted_jaccard=True
- low_df_threshold, min_low_df_anchors, idf_mean_min

Candidate limits:
- min_shared_shingles, size_ratio_[min,max], max_candidates_per_block
- Optional knobs (off by default): enable_hybrid_bandset [+ bandset_*], enable_mutual_topk_filter [+ mutual_k]

Pruning/community:
- degree_cap, [k_core_min_degree], [triangle_support_min], use_community_detection

---

## 14. Recommended Profile (Default)

- Strict alignment gate: min_anchors=5; min_span_genes=10; v_mad_max_genes=1.
- Strand-insensitive tokens (null last embedding dim before SRP).
- XOR shingling with k=3 and skip-gram pattern [0,2,5]; contiguous if unset.
- Weighted Jaccard with df_max=200; low-DF anchor and IDF-mean checks.
- No hybrid augmentation.
- Post-cluster attach (PFAM-agnostic): exp8-like thresholds; stitches tiny sink blocks when strongly supported; cluster count unchanged; purity preserved.

---

## 15. Diagnostics and Telemetry (Δ)

- Print counts: |𝓡|, sink, |unique shingles|, |edges|, |clusters|.
- Optionally print DF histograms and top shingles by DF.
- Sanity: count 6/6 RP clusters by genome coverage in syntenic_blocks.csv.

---

## 16. Validation & Diagnostics (V)

- tools/diagnose_block_vs_cluster.py: explains why a block lacks edges to a target cluster (overlap pre/post-DF, bandset J, order modes).
- tools/evaluate_curated_rp_purity.py: reports non-sink cluster count and curated-RP cluster purity/size.
- tools/check_canonical_rp_alignment.py: flags canonical RP loci aligning to non-canonical partners (should be 0 after alignment strictness).
- tools/attach_by_cluster_signatures.py: post-cluster attach (exp8 thresholds), preserving compactness and purity.

---

## 17. Pitfalls and Anti-Patterns (π)

- XOR k-grams can 0-out under small band flips even with high cosine; hybrid channel fixes this for conserved operons.
- Setting df_max too low starves edges; too high introduces hubs (use degree cap + IDF + min_low_df_anchors).
- Over-aggressive pruning (min_shared, caps, mutual-only) can collapse clusters (e.g., drop to ~20); tune caps conservatively.

---

## 18. Future Work (Φ)

- Parallel scoring with `ProcessPoolExecutor` behind `jobs` (configurable).
- Exact mutual-k with reciprocal top-k lists for more stability with similar performance.
- Alternative communities: Leiden/Louvain (igraph) if NetworkX becomes a bottleneck.
- Learnable thresholds via small meta-optimization loop over validation sets.

---

## 19. Tuning Cheat-Sheet (TC)

Symptoms → Actions:
- “RP missing 2 genomes” → lower bandset_tau (0.20), raise bandset_df_max (5000), min_shared_* = 1 temporarily.
- “Too many clusters” → add enable_mutual_topk_filter; raise min_shared_shingles to 3; tighten degree_cap.
- “Runtime high post-DF” → increase min_shared_shingles; lower max_candidates_per_block; enable mutual-top-k; reduce bandset_topk_candidates.

---

## 20. Reproducibility Fingerprint (RF)

Emit JSON:
```
{
  srp: {bits, bands, band_bits, seed, hash(R)},
  df: {df_max, bandset_df_max, max_df_percentile},
  gates: {min_anchors, min_span_genes, v_mad_max_genes, cassette_max_len},
  shingles: {method, k, bands_per_window, band_stride},
  hybrid: {enabled, tau, min_len, min_identity},
  prune: {min_shared_shingles, max_candidates_per_block, bandset_topk_candidates},
  graph: {degree_cap, mutual_topk, mutual_k},
  dataset: {n_windows, n_blocks, robust: |𝓡|},
  telemetry: {unique_shingles, edges, clusters}
}
```

---

## 21. Implementation Pointers

- Python sets for postings and intersections; tune caps to retain O(Σ|C_b|) linear behavior.
- IDF arrays as dicts keyed by int tokens; avoid NumPy overhead for sparse sums.
- Deterministic ordering for sorting candidates and components.
- Avoid per-block logging inside tight loops.

---

## 22. Glossary

- SRP: Signed Random Projection; binary sketches from hyperplane tests in embedding space.
- Shingle: fixed-length k-gram over window tokens (order-aware); bandset: order-agnostic union of band tokens.
- DF/IDF: document frequency and inverse document frequency over shingles/band tokens.
- Mutual-k: reciprocal top-k neighborhood condition (optional/approximate here).
- Degree cap: keep top-N edges incident to a node by weight.

---

## 23. Status

- Default profile: strand-insensitive tokens + XOR k=3 skip-gram [0,2,5], weighted Jaccard with df_max=200, no hybrid augmentation, post-attach enabled.
- Diagnostics tools in `tools/` help validate compactness and RP purity.

[EOF]
