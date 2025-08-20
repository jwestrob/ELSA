# ELSA Syntenic Block Clustering — Technical Design (For Model Review)

This document describes the clustering subsystem that assigns syntenic blocks to cassette families using mutual-Jaccard over SRP-derived shingles, with an optional hybrid augmentation. It is written in a maximally dense, implementation-proximal style designed for iterative refinement and machine review.

---

## 1. Objectives and Scope

- Group collinear syntenic blocks into “cassette” families across genomes.
- Leverage continuous embeddings (SRP) to obtain order-aware, orientation-stable token sequences.
- Use a sparse similarity graph with conservative pruning; detect communities via modularity.
- Preserve determinism and re-runnability across machines and seeds.
- Optional: augment edges for long, high-identity operons where XOR shingling is brittle.

Non-goals: Inferring evolutionary history; gene-by-gene orthology; plasmid-level topology.

---

## 2. Data Model

Block b has:
- `query_windows`, `target_windows`: ordered IDs of windows in the collinear chain.
- `matches`: list of paired windows (query_window_id, target_window_id) used for robustness gating.
- `alignment_length` (|matches|), `identity` (mean match similarity), `strand` (+1 or -1).

Windows W have embeddings E ∈ ℝ^D (post-PLM projection). Per-sample, windows are indexed in Parquet.

---

## 3. Robustness Gate (R)

Filter blocks before tokenization:
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

### ICWS-r window sketch (tokenizer.mode = icws)
We replace XOR band collapsing with r (default 8) ICWS samples per window. Each sample selects one band-id with collision probability proportional to its weight (currently uniform). The ordered r-tuple is serialized as the window token and consumed by shingling.

- Skip-gram shingling: Enable k=3 skip-grams using offsets like (0,2,5) via `shingle_pattern`. This increases robustness to small insertions/deletions while preserving order/orientation semantics. If `shingle_pattern` is not set, contiguous k-grams are used (legacy behavior).
- Parameters: `shingle_method ∈ {xor, subset, bandset, icws}`, `icws_r`, `icws_bbit`, `shingle_pattern`.
- Determinism: A global seed (`srp_seed`) deterministically derives per-window/per-sample sub-seeds.

Migration: keep `shingle_method: xor` to reproduce legacy behavior. To try ICWS while preserving defaults elsewhere, set `shingle_method: icws` and optionally `shingle_pattern: "0,2,5"`.

---

## 6. DF Filtering + IDF (F)

- Build inverted index postings for shingles: P[s] = {b | s ∈ S_b}.
- Compute DF(s) = |P[s]|.
- Filter S_b ← {s ∈ S_b | DF(s) ≤ `df_max`}.
- Rebuild postings P′ from filtered sets.
- IDF(s) = log(1 + N/DF(s)) with N = |{b | S_b ≠ ∅}|.

Optionally ban top-percentile DF (max_df_percentile), then rebuild.

For hybrid bandset: repeat DF/IDF on band tokens with looser `bandset_df_max` (Section 9).

---

## 7. Candidate Generation (C)

For each block b with filtered S_b:
- Accumulate candidate counts via postings: c[x] = |S_b ∩ S_x| using P′.
- Keep x with c[x] ≥ `min_shared_shingles`.
- Size prefilter: let s_b = |S_b|, s_x = |S_x|; and alignment lengths n_b, n_x. Accept if `size_ratio_min` ≤ s_b/s_x ≤ `size_ratio_max` and same for n_b/n_x.
- Keep top-`max_candidates_per_block` by c[x].

Hybrid augmentation (qualified b): add up to `bandset_topk_candidates` candidates x by shared band tokens ≥ `min_shared_band_tokens`, with same size prefilters. Qualification: n_b ≥ `bandset_min_len` and identity ≥ `bandset_min_identity`.

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

Optional local cap: if `enable_mutual_topk_filter` true, restrict to top-`mutual_k` candidates per b before materializing edges (approximate mutual-k gating). Full symmetric mutual-k can be added if needed (requires storing per-node top lists and intersecting).

---

## 9. Hybrid Bandset Augmentation (H)

For long, highly conserved operons, XOR/subset shingles can 0-out despite high cosine and near-identity. The bandset channel recovers a weak but robust signal.

For qualified pairs (b,x):
- B = bandset(b), X = bandset(x) after DF filtering with `bandset_df_max`.
- Weighted bandset Jaccard:
  J_bw(b,x) = Σ_{t ∈ B∩X} IDF_b(t) / Σ_{t ∈ B∪X} IDF_b(t).
- If no shingle-edge was added and J_bw ≥ `bandset_tau` ⇒ add edge with weight = J_bw.

This yields recall for conserved loci (e.g., ribosomal operons) with small runtime impact due to tight qualification and candidate bounds.

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
- bandset_df_max (hybrid)

Similarity and edges:
- jaccard_tau, use_weighted_jaccard=True
- low_df_threshold, min_low_df_anchors, idf_mean_min

Candidate limits:
- min_shared_shingles, size_ratio_[min,max], max_candidates_per_block
- enable_hybrid_bandset, bandset_min_len, bandset_min_identity, min_shared_band_tokens, bandset_topk_candidates, bandset_tau
- enable_mutual_topk_filter, mutual_k

Pruning/community:
- degree_cap, [k_core_min_degree], [triangle_support_min], use_community_detection

---

## 14. Recommended Profiles (R*)

- control (fastest, brittle): xor + k=3; df_max=200; no hybrid; min_shared_shingles=1; no mutual-top-k; yields partial RP (≤5/6).
- hybrid_mid (balanced; default): enable_hybrid_bandset; bandset_tau=0.25; bandset_df_max=3000; (min_len=20, id≥0.98); min_shared_shingles=2; min_shared_band_tokens=2; caps: 2000/500; degree_cap=10. Recovers RP 6/6.
- hybrid_loose (max recall): as above with bandset_tau=0.20, bandset_df_max=5000, min_shared=1; slightly larger graphs, still fast on test set.
- hybrid_mutual (fewer edges): add enable_mutual_topk_filter with mutual_k=3; may reduce extra 6/6 duplicates.

---

## 15. Diagnostics and Telemetry (Δ)

- Print counts: |𝓡|, sink, |unique shingles|, |edges|, |clusters|.
- Optionally print DF histograms and top shingles by DF.
- Sanity: count 6/6 RP clusters by genome coverage in syntenic_blocks.csv.

---

## 16. Validation Harness (V)

- tools/test_rp_cluster.py: A/B profiles; asserts presence of a 6-genome RP cluster.
- tools/experiment_rp_params.py: grid search; reports runtime, cluster counts, and best RP coverage.
- Quick CSV check: count non-sink clusters and coverage sets pre-browser.

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

- Current default: `hybrid_mid` operational; passes RP-6/6 sanity on test dataset.
- Control profile documented for A/B.
- Experiment harness in tools/ supports rapid param exploration.

[EOF]
