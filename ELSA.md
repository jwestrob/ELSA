# ELSA — Embedding Locus Shingle Alignment
Order-aware syntenic-block discovery from protein language-model embeddings.

This document is the implementation blueprint for a standalone ELSA package that our agent (and external users) can call. It includes ingestion from FASTA, configurable PLM embeddings (ESM2 or ProtT5), and preparation of the dual index required for automated locus/block discovery.

---

## 0) Goals and constraints

**Primary capability.**
Given assembled genomes or single metagenomes, compute protein embeddings, construct order-aware shingle embeddings and dual recall indexes (discrete MinHash on codeword n-grams + continuous signed-random-projection on window embeddings), then expose high-recall, high-precision discovery of similar syntenic blocks via collinearity chaining.

**Compute envelope.**
Single machine, CPU-first; runs comfortably on a 48 GB RAM Apple M-series laptop. GPU optional for faster PLM embedding but not required. All heavy arrays memory-mapped; batchable and resumable.

**Data scope.**
A batch of individual genomes or a small number of metagenomes. We avoid pangenome-scale cross-sample joins in v1; the design remains shardable.

---

## 1) User-facing interfaces

### 1.1 CLI surface
- ~elsa init~ → write a starter ~elsa.config.yaml~ and ~samples.tsv~ template.
- ~elsa embed~ → FASTA → gene calls → AA FASTA → PLM embeddings → projected Parquet.
- ~elsa build~ → build discrete and continuous indexes from projected embeddings.
- ~elsa find~ → find syntenic blocks for a query locus against a chosen scope.
- ~elsa explain~ → show top shingles and aligned window pairs for a block.
- ~elsa stats~ → QC summaries (collisions, IDF spectrum, anchor recall sanity).

### 1.2 Service API (FastAPI; agent-friendly)
- ~POST /build/embed~  payload: ~{samples_tsv, config_path}~  → status, artifact registry id.
- ~POST /build/index~  payload: ~{registry_id or paths}~ → status, index id.
- ~POST /find_like_block~ payload: ~{query_locus, target_scope, params?}~ → blocks.
- ~GET /explain/{block_id}~ → anchors, shingles, local alignments, scores.

All responses are compact JSON; schemas defined in section 6.

---

## 2) Configuration

Single canonical file: ~elsa.config.yaml~. Keys are validated and frozen into the artifact registry.

Template (values are defaults, not data):

- ~data:~
  - ~samples_tsv: ./samples.tsv~
  - ~work_dir: ./elsa_index~
  - ~allow_overwrite: false~

- ~ingest:~
  - ~gene_caller: prodigal~          # prodigal|metaprodigal|none (if AA+GFF provided)
  - ~prodigal_mode: single~
  - ~min_cds_aa: 60~
  - ~keep_partial: false~

- ~plm:~
  - ~model: prot_t5~                  # prot_t5|esm2_t33|esm2_t12
  - ~device: auto~                    # cpu|cuda|mps|auto
  - ~batch_amino_acids: 16000~        # approximate per-batch residue budget
  - ~fp16: true~
  - ~project_to_D: 256~               # PCA target dim D
  - ~l2_normalize: true~

- ~shingles:~
  - ~n: 5~
  - ~stride: 1~
  - ~pos_dim: 16~
  - ~weights: triangular~             # triangular|uniform|gaussian
  - ~strand_flag: signed~             # signed|onehot

- ~discrete:~
  - ~K: 4096~                         # codebook centroids
  - ~minhash_hashes: 192~
  - ~bands_rows: [24, 8]~             # bands × rows
  - ~emit_skipgram: true~
  - ~idf_min_df: 5~
  - ~idf_max_df_frac: 0.05~

- ~continuous:~
  - ~srp_bits: 256~
  - ~srp_seed: 13~
  - ~hnsw_enable: false~

- ~chain:~
  - ~offset_band: 10~                 # Δ windows allowed drift
  - ~gap_open: 2.0~
  - ~gap_extend: 0.5~
  - ~slope_penalty: 0.05~

- ~dtw:~
  - ~enable: true~
  - ~band: 10~

- ~score:~
  - ~alpha: 1.0~   # anchor strength
  - ~beta: 0.1~    # LIS length
  - ~gamma: 0.2~   # gap penalty
  - ~delta: 0.1~   # offset variance penalty
  - ~fdr_target: 0.01~

- ~system:~
  - ~jobs: auto~
  - ~mmap: true~
  - ~rng_seed: 17~

---

## 3) Input inventory

### 3.1 samples.tsv
Tab-delimited manifest; one dataset per row.

Columns:
- ~sample_id~: unique string
- ~fasta_path~: nucleotide FASTA of assembled contigs
- ~gff_path~: optional; if present, ingester uses features instead of calling genes
- ~aa_fasta_path~: optional; if present, skips translation if consistent with GFF
- ~emb_path~: optional; precomputed PLM embeddings for proteins (Parquet or NPZ)
- ~contig_whitelist~: optional comma-sep list
- ~notes~: free text

Rules:
- If ~emb_path~ supplied, projection will read from there; otherwise ~elsa embed~ computes embeddings from AA sequences produced or provided.

---

## 4) Build pipeline (end-to-end)

### Stage A — Ingest
- Parse FASTA; index contigs and lengths.
- If ~gff_path~ provided: extract CDS features (start, end, strand, phase), enforce monotone ordering; filter short CDS < ~min_cds_aa~; translate to AA FASTA using the specified code (default 11).
- If no GFF: call genes with Prodigal or MetaProdigal; generate GFF + AA FASTA.
- Emit ~genes_raw.parquet~: ~{sample_id, contig_id, gene_id, start, end, strand:int8, aa_len:int, aa_offset:int64, aa_len_bytes:int32}~ plus offsets into a compact AA blob.
- Emit ~proteins.faa~ (optional convenience).

### Stage B — PLM embeddings
- Load chosen PLM (ProtT5 or ESM2 variant) according to ~plm.model~ and ~plm.device~.
- Stream AA sequences in length-balanced batches (respect ~batch_amino_acids~). Tokenize, forward pass, pool to per-protein embeddings (CLS or mean over residues; configurable but default mean-pool).
- Write ~emb_raw.parquet~: ~{sample_id, gene_id, E_raw: float16[de or proj], de:int16}~ as chunked binary; optionally store separate NPZ shards per sample if preferred by the runtime.

### Stage C — Projection & normalization
- Fit PCA to target ~project_to_D~ on a streamed subsample (stratified by length and contig) unless user provides a frozen PCA model.
- Transform all embeddings to ℝ^D, L2-normalize if ~l2_normalize: true~; write ~genes.parquet~:
  - ~{sample_id, contig_id, gene_id, start, end, strand, E: float16[D]}~
- Persist ~pca_model.npz~ metadata for reproducibility.

### Stage D — Shingling (window embeddings)
- Slide windows of size ~n~ with stride ~1~ over each contig’s ordered genes.
- Build order-aware window vector ~x_j~ by concatenating each gene vector with positional encoding of the offset and a strand indicator, linearly projecting to ℝ^D (tiny learned or PCA-fit projection; in v1 we reuse PCA for simplicity).
- Emit ~windows.parquet~: ~{sample_id, locus_id, j:int32, E_w: float16[D], sigma:int8}~ where ~sigma~ is the strand sign of the window.

### Stage E — Discrete index (codewords → n-grams → MinHash)
- Train MiniBatch KMeans with ~K~ centroids on a large sample of protein vectors ~E~.
- Assign each gene its nearest codeword id; write ~codewords.parquet~ ~{gene_id, cw:uint16}~.
- Emit orientation-aware n-grams (and one skip-gram per window if ~emit_skipgram~) as uint64 hashes into ~shingles.parquet~ ~{locus_id, j, sigma, ngram_hash:uint64}~.
- Compute per-shingle document frequencies to create ~idf.parquet~ ~{ngram_hash, df:int32, idf:float32}~ with the configured min/max df thresholds.
- Build MinHash signatures with ~minhash_hashes~ and store band keys in ~minhash/part-*.parquet~ ~{locus_id, band:int16, key:uint64}~.

### Stage F — Continuous index (SRP)
- Sample SRP hyperplanes in ℝ^D with ~srp_seed~ to get ~srp_bits~ sign bits.
- For each window embedding ~E_w~, compute its 256-bit signature and write packed ~uint64[4]~ to ~srp.parquet~ ~{locus_id, j, sig0, sig1, sig2, sig3}~.
- Build band buckets (e.g., eight 32-bit slices) for fast lookup.

### Stage G — Registry and QC
- Create ~MANIFEST.json~ capturing:
  - config hash, codebook hash, PCA hash, SRP seed, dataset stats.
- Produce ~qc/*.pdf~ summarizing IDF spectrum, codeword usage, MinHash collision rates, SRP bucket occupancy, and synthetic shuffle recall.

At this point the database is prepared for locus/block identification.

---

## 5) Find-time pipeline

- **Recall.** For each query window j: pull top candidates from the discrete LSH (Jaccard via MinHash on shingles near j) and the continuous SRP buckets (Hamming-nearest; then cosine on ~E_w~ for top-K). Union and deduplicate.
- **Anchor scoring.** Score anchor ~a=(j→j~prime)~ with ~s_a = λ1·cos(E_w[j], E_w[j~prime]) + λ2·J_local + λ3·IDF_sum~; non-max suppression per j to control density.
- **Chaining.** Sort anchors by target coordinate; run weighted LIS with affine gaps in an offset band of size ~offset_band~, for both strands. Backtrack to get chains.
- **Refinement (optional).** Banded DTW with bandwidth ~dtw.band~ on the matched window sequences to sharpen boundaries and reject spurious chains.
- **Score & emit.** Compute final ~S = α·Σ s_a + β·LIS_len − γ·gaps − δ·offset_var~; calibrate τ for ~fdr_target~ via within-locus shuffles, then emit ~blocks.jsonl~.

---

## 6) Data models and file schemas

### 6.1 Parquet: genes.parquet
Columns:
- ~sample_id:str, contig_id:str, gene_id:str~
- ~start:int32, end:int32, strand:int8~
- ~E: fixed_size_list<float16, D>~

### 6.2 Parquet: windows.parquet
- ~sample_id, locus_id:str, j:int32, sigma:int8~
- ~E_w: fixed_size_list<float16, D>~

### 6.3 Discrete index
- ~codebook.npz~: ~C[K, D]: float32~, metadata
- ~codewords.parquet~: ~{gene_id, cw:uint16}~
- ~shingles.parquet~: ~{locus_id, j, sigma, ngram_hash:uint64}~
- ~idf.parquet~: ~{ngram_hash:uint64, df:int32, idf:float32}~
- ~minhash/*.parquet~: ~{locus_id, band:int16, key:uint64}~

### 6.4 Continuous index
- ~srp_hyperplanes.npz~: ~H[D, srp_bits]: float32~
- ~srp.parquet~: ~{locus_id, j:int32, sig0:uint64, sig1:uint64, sig2:uint64, sig3:uint64}~

### 6.5 Outputs
- ~blocks.jsonl~ entries:
  - ~block_id, sample_q, locus_q, q_start, q_end, sample_t, locus_t, t_start, t_end, strand~
  - ~anchors: [{jq:int, jt:int, score:float32}]~
  - ~score:float32, dtw:{cost:float32, steps:int32, band:int32}~
  - ~signature:{top_ngrams:[uint64], centroid:float32[D_small]}~

### 6.6 API schemas (pydantic)
- Request ~FindLikeBlockRequest~: ~{query_locus:str, target_scope:str, params:Optional[dict]}~
- Response ~Block~ mirrors ~blocks.jsonl~ item.

---

## 7) File tree (after ~elsa build~)

- ~elsa_index/~
  - ~MANIFEST.json~
  - ~params.lock.yaml~
  - ~ingest/~
    - ~genes_raw.parquet~
    - ~proteins.faa~                # optional
    - ~genes.parquet~               # projected ℝ^D, float16
    - ~pca_model.npz~
  - ~shingles/~
    - ~windows.parquet~
    - ~codebook.npz~
    - ~codewords.parquet~
    - ~shingles.parquet~
    - ~idf.parquet~
    - ~minhash/part-*.parquet~
  - ~srp/~
    - ~srp_hyperplanes.npz~
    - ~srp.parquet~
  - ~qc/*.pdf~

---

## 8) Determinism, resumption, and resource knobs

- All random choices (PCA randomized SVD, MiniBatch KMeans seeding, SRP hyperplanes) are seeded from ~system.rng_seed~ and recorded in ~MANIFEST.json~.
- ~jobs: auto~ maps to physical cores; can be overridden per command.
- Memory bounded by float16 storage, memory-mapped reads, and per-stage batch sizes.
- Every stage writes checkpoints; ~elsa build~ is idempotent and can resume.

---

## 9) Testing and validation plan

- **Unit tests**: GFF parsing edge cases; Prodigal round-trip; PCA projection shape; SRP signature stability; MinHash banding determinism.
- **Property tests**: chaining invariance to permutation of equal-score anchors; DTW monotonic path constraints; Jaccard/MinHash unbiasedness on synthetic shingle sets.
- **Metamorphic**: jitter gene boundaries ±3–6 aa; results should be stable within tolerance thresholds.
- **Golden set**: hand-curated loci with known duplications or BGC variants; expected block ranges and high-level scores pinned.

---

## 10) Claude-Code workplan (sequenced)

1. **Skeleton**: create ~elsa/~ package, ~cli.py~, ~api/service.py~, ~params.py~ with strict validation.
2. **IO**: implement ~manifest.py~, ~parse_gff.py~, ~embeddings.py~ readers and streaming AA accessor.
3. **Embedding**: hook ProtT5 and ESM2 loaders; implement batcher and mean-pooling; write ~emb_raw.parquet~.
4. **Projection**: PCA fit/transform module; write ~genes.parquet~; lock model.
5. **Shingling**: window generator with positional encodings and strand flags; ~windows.parquet~ writer.
6. **Discrete path**: codebook training, quantization, n-gram hashing, IDF, MinHash banding, LSH lookup.
7. **Continuous path**: SRP hyperplanes, signatures, band buckets; Hamming search primitive.
8. **Search**: anchor union, scoring, NMS; chaining DP; optional DTW refine; final scorer.
9. **CLI**: wire ~init~, ~embed~, ~build~, ~find~, ~explain~, ~stats~ with robust logs and progress bars.
10. **Service**: minimal FastAPI with ~/find_like_block~ and ~/explain~; streaming responses for large result sets.
11. **QC**: plotting utilities for IDF spectrum, collisions, bucket occupancy; write ~qc/*.pdf~.
12. **Docs**: ~docs/overview.md~, ~docs/cli.md~, ~docs/file_formats.md~, ~docs/api.md~ with examples pulled from tests.

---

## 11) Open options and v1 decisions

- Window projection ~W~: v1 uses PCA only; v1.1 may add a small learned linear layer trained to preserve cosine distances among neighboring windows.
- HNSW: off by default; enable only for large corpora where SRP recall requires re-ranking.
- Protein pooling: default mean-pool; allow CLS for ESM2 via a param.
- Gene caller: default Prodigal single; ~metaprodigal~ optional for low-N50 metagenomes.

---

## 12) Acceptance criteria for v1

- ~elsa embed~ completes on 10–30 M genes within the 48 GB envelope, producing ~genes.parquet~ and QA plots.
- ~elsa build~ constructs dual indexes and reports collision and bucket stats within expected bands.
- ~elsa find~ returns blocks in sub-second to a few seconds per query locus on CPU, with stable scores across reruns.
- ~elsa explain~ provides human-readable shingles and aligned window pairs.

---

## 13) License and citation

- Apache-2.0 or MIT for code, model weights deferred to upstream licenses (ProtT5, ESM2).
- Method name: **ELSA: Embedding Locus Shingle Alignment**. Prepare a short methods note when results stabilize.

---

