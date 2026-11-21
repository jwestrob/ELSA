# ws_tokens notebook (append-only)

Workstream: family tokens for the recurrence-first ELSA redesign. Brief:
`docs/redesign/BRIEF.md`. Remote: `$RD/ws_tokens/`, finished arms in `$RD/tokens/`.

---

## 2026-11-11 04:15 PST: plan and predeclared criteria (written before any token or result exists)

### Question

Can a discrete token per protein, from k-means cells of a protein language-model embedding,
stand in for a homologous family in a recurrence engine? The comparison arm is MMseqs2
clustering at >= 30% identity and >= 80% coverage of both sequences
(`easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0`, the rule of the 2 November DPANN pass,
`/groups/banfield/users/jwestrob/dpann_dark_modules_2026-11-02/run.sbatch`).

### Terms

- *same-token pair*: two genes in different genomes with the same token >= 0.
- *pair recall* for a label set: fraction of its pairs whose genes share a token. A pair with
  a token of -1 on either side counts as a miss; denominators are all pairs of the set.
- *sequence precision*: among sampled same-token pairs, the fraction with MMseqs2
  Smith-Waterman E <= 1e-3 (pairwise, `-e inf --alignment-mode 3`).
- *homology precision*: sequence precision + (1 - sequence precision) x the fraction of
  sampled sequence-invisible same-token pairs with TM >= 0.5 (normalized by the shorter
  chain, i.e. max(qTM, tTM)). TM >= 0.5 means shared fold, and is reported as that.
- *collision rate*: pair recall on the 300,000 random query x reference pairs.

### Predeclared readouts

**Against embedding tokens as a family stand-in.** Embedding tokens are a poor substitute
for families if, for every model and transform tested, at every K in {65,536, 262,144,
1,048,576}, either
- (F1) pair-uniform sequence precision < 50%, or
- (F2) recall on sequence-visible native Spacedust pairs < 50%, or
- (F3) collision rate > 0.5% (tokens too coarse to carry family identity).

**For embedding tokens over sequence clusters.** An embedding arm is better than the
sequence arm if all of:
- (B1) recall on sequence-invisible in-block pairs with TM >= 0.5 exceeds the MMseqs arm's
  by >= 0.10 absolute, with non-overlapping 95% Wilson intervals;
- (B2) homology precision >= 0.9 x the MMseqs arm's homology precision;
- (B3) recall on sequence-invisible in-block pairs with TM < 0.5 is at most half its recall
  on TM >= 0.5 pairs (the tokens discriminate folds, rather than merging everything);
- (B4) collision rate <= 0.1%.

**Expected direction (stated so a surprise is visible).** Larger K raises precision and
lowers recall. Genome-centring removes a per-genome offset; if lineage signal is part of
what joins orthologs, genome-centring lowers recall on Spacedust pairs, and if it is noise,
it raises V-measure against the MMseqs arm.

**Recommendation rule for the engine (predeclared).** Among embedding arms with homology
precision >= 80% and collision rate <= 0.1%, recommend the one with the highest recall on
sequence-invisible TM >= 0.5 in-block pairs, plus the MMseqs arm as the baseline. If no
embedding arm meets both gates, recommend the MMseqs arm alone and say so.

### Plan and order

1. `kmeans__esm2_8m__raw__K262144` (then K65536 and K1048576 from the same coarse codebook).
2. `mmseqs__seq__id30c80`: one database of all 4,257,531 proteins keyed by gid (sequences
   extracted from `querydb` and `keggclusterdb`, lengths checked against the universe for
   every row), `mmseqs cluster` with the DPANN rule.
3. Label pair sets in gid space (Spacedust, ESM2-650M accept05 blocks, TM key set, random).
4. ESM2-8M grid: {raw, global-centred, genome-centred} x K.
5. Models esm2_650m, esmc600m, esmc6b at raw K262144 plus the two best settings from step 4.
6. If time remains: mutual-kNN graph communities.

### Method decisions fixed in advance

- Two-level spherical k-means. Coarse: 4,096 cells, faiss spherical k-means, 20
  iterations, trained on 409,600 vectors sampled with seed 17, all vectors assigned by
  inner product. Fine: in each cell, k_c = max(1, round(K x n_c / N)) (capped at n_c),
  spherical Lloyd iterations (25 at most, stop when assignments are unchanged), random
  initialization from the cell's points with seed 17 x 1,000,003 + cell. Final tokens are
  renumbered densely over the non-empty fine centroids.
- Pair precision samplers, 200,000 pairs each, seed 17, with replacement:
  pair-uniform (token drawn with probability proportional to its number of cross-genome
  pairs, then a uniform cross-genome pair inside it; this is uniform over all same-token
  cross-genome pairs) and token-uniform (token drawn uniformly among tokens with at least
  one cross-genome pair, then a uniform cross-genome pair).
- Pair alignment runs on the combined gid-keyed database on both sides. Its size is
  4,257,531 proteins against 4,190,383 for `keggclusterdb`, so E-values are about 1.6%
  larger than with the reference alone; this is the only departure from `$S/code/pipeline.sh`.
- V-measure against the MMseqs arm on all genes with a token in both arms (the brief allows
  a 1,000,000-gene sample; the full set is used if it fits, and the number used is stated).

## 2026-11-11 04:40 PST: first two jobs submitted

Preflight: read `/home/jwestrob/.claude/CLAUDE.md` and
`/home/jwestrob/jwestrob/obsidian-vault/biotite-cluster-ops.md` in full before the sbatch.

Code (local `code/`, remote `$RD/ws_tokens/code/`): `common.py` (paths, atomic token writer,
INDEX append under flock), `kmeans_tokens.py`, `make_fasta.py`, `mmseqs_tokens.py`,
`run_kmeans.sbatch`, `run_mmseqs.sbatch`.

Change to the predeclared method, before any result: fine-level seeding is k-means++ on the
sphere (squared chord distance), in place of random points. On a synthetic 30-cluster test
(20,000 points, 32 D) V-measure rose from 0.938 (random seeding) to 0.970-0.980.

| job | what | resources |
|---|---|---|
| 76252 | `run_kmeans.sbatch esm2_8m raw 262144,65536,1048576` | standard_expand, 24 CPU, node-64-768g-30 |
| 76253 | `run_mmseqs.sbatch` (FASTA by gid, createdb --shuffle 0, cluster 0.3/0.8/cov-mode 0) | standard_expand_2, 64 CPU, node-64-768g-10 |

## 2026-11-11 04:15 MST: correction to timestamps

The two entries above are stamped "PST" but were written at about 04:05 and 04:12 MST (Mac
clock). From here on every stamp is MST (Mac clock); biotite logs print PST, one hour
earlier. Jobs 76252 and 76253 started at 03:12:39 PST. Query FASTA extraction check:
67,148 rows, 0 missing keys, 0 length mismatches.

## 2026-11-11 04:25 MST: labels built; k-means fine stage moved to processes

**Labels** (`labels.py`, job 76257 then 76265 after a `tsv2db --threads` flag error in the
foldseek alias step; the label step itself succeeded both times in 3.6 s):
`$RD/ws_tokens/labels/label_pairs.parquet`, `label_stats.json`.

| set | pairs |
|---|---:|
| sd_vis (native Spacedust, SW E <= 1e-3) | 3,955,569 |
| sd_invis | 403,491 (9.3% of 4,359,060, as in the brief) |
| blk650_vis (accept05 ESM2-650M blocks) | 759,235 |
| blk650_invis | 293,345 |
| random | 300,000 |
| tm_hi / tm_lo | pending: `$S/work_key/tm_key.parquet` not yet written (job 76209 running) |

Checks: all 4,359,060 Spacedust pairs have a row in `aln_seq.tsv`; all 1,052,580 ESM2-650M
accept05 block pairs exist in that model's `anchors.parquet` (trap 2 check); every label pair
mapped to gids through the universe (trap 1).

Combined FASTA (`make_fasta.py`): 4,257,531 sequences, 1,394,754,437 residues, 0 missing
keys and 0 length mismatches against the universe on both sides.

Foldseek alias databases with key headers (`K<key>` reference, `Q<key>` query):
`$RD/ws_tokens/work/fsdb/{rdb,qdb}`.

**Correction to the k-means run.** Jobs 76252 (raw), 76262 (globalcentred), 76263
(genomecentred) were cancelled after 5.5 / 1.3 / 1.3 min: the fine stage ran per-cell numpy
in a thread pool and used 2.2 of 24 cores (GIL-bound on small matrices). The fine stage now
runs in a fork-based process pool (cells >= 20,000 genes would run in the parent with all
BLAS threads; ESM2-8M raw has none, maximum cell 4,762). Results are deterministic (checked
on synthetic data). Resubmitted as 76270 (raw), 76271 (globalcentred), 76272
(genomecentred), K = 262,144, 65,536, 1,048,576 in that order.

Observation from the cancelled raw run: coarse codebook trained in 125 s; mean cosine of a
vector to its coarse centroid 0.981; cell sizes 1 / 958 / 4,762 (min / median / max).
The global mean of the raw ESM2-8M vectors has norm 0.894: most of each unit vector is a
shared offset, consistent with trap 4 (unrelated proteins near cosine 0.93).

## 2026-11-11 04:25 MST: arm 1 delivered

`$RD/tokens/kmeans__esm2_8m__raw__K262144.parquet` written 03:21:28 PST by job 76270:
262,158 tokens, 4,256,595 genes assigned, 936 unassigned (the query proteins > 1,022 aa
without an ESM2-8M vector), 155 s total (137 s load + coarse, 18 s fine), 24 CPUs, peak RSS
10.3 GB. Fine stage: k_c median 59 (max 293); 48 of 4,096 cells stopped at the 25-iteration
cap; the rest converged (median 13 iterations).
`kmeans__esm2_8m__genomecentred__K262144` was written at the same time by job 76272
(262,158 tokens; coarse cos to centroid 0.875 after centring, against 0.981 raw).

Code provenance: every manifest carries `code_sha256` over the code directory;
`$RD/ws_tokens/code_snapshots/<hash>/` holds the matching files (the writer now snapshots
automatically; 759ab600... and bf97f9a5... were snapshotted by hand).

Added `graph_tokens.py` (step 5; not yet run) and a workstream venv `$RD/ws_tokens/env`
(system site packages + igraph 0.11.8, leidenalg 0.10.2) for it.

## 2026-11-11 04:30 MST: ESM2-8M grid tokens complete; quality job queued

All nine ESM2-8M arms are in `$RD/tokens/` (jobs 76270-76272, 3.3-3.7 min each, 24 CPUs):

| arm | tokens | seconds |
|---|---:|---:|
| kmeans__esm2_8m__raw__K65536 | 65,550 | 152.9 |
| kmeans__esm2_8m__raw__K262144 | 262,158 | 155.0 |
| kmeans__esm2_8m__raw__K1048576 | 1,048,569 | 161.9 |
| kmeans__esm2_8m__globalcentred__K65536 | 65,531 | 171.3 |
| kmeans__esm2_8m__globalcentred__K262144 | 262,164 | 172.6 |
| kmeans__esm2_8m__globalcentred__K1048576 | 1,048,582 | 181.3 |
| kmeans__esm2_8m__genomecentred__K65536 | 65,539 | 147.4 |
| kmeans__esm2_8m__genomecentred__K262144 | 262,158 | 148.3 |
| kmeans__esm2_8m__genomecentred__K1048576 | 1,048,601 | 157.2 |

(Seconds include the shared load + transform + coarse stage, about 135-165 s, which runs once
per transform.) Genome means of ESM2-8M vectors have norm 0.880 / 0.909 / 0.962
(min / median / max over 1,317 genomes).

Submitted, not yet run: 76276 (`esm2_650m raw`, all three K), 76277 (`esmc600m raw`, all three K),
and quality job 76284 (`run_quality.sbatch grid8m <mmseqs + 9 ESM2-8M arms> 1`, TM on), with
`--dependency=afterok:76253:76270:76271:76272`. TM-align runs on every arm of the grid:
20,000 pairs per arm cost about 1.5 min at 24 CPUs, so ranking first is unnecessary.

Housekeeping: the scratchpad `watch.sh` was overwritten by another session sharing the
scratchpad; my watchers now use `code/local/tok_watch.sh` in this directory.

## 2026-11-11 04:37 MST: arm 2 delivered (MMseqs2 sequence families)

`$RD/tokens/mmseqs__seq__id30c80.parquet` written 03:25 PST by job 76253:
**1,038,303 clusters** over all 4,257,531 genes (token = representative gid). `mmseqs cluster
--min-seq-id 0.3 -c 0.8 --cov-mode 0` (MMseqs2 release 18, 8cc5ce36), cascaded, 692 s wall on
64 CPUs (5,481% CPU), peak RSS 6.5 GB. Verified: lookup key == gid for all 4,257,531 rows;
every gid appears once as a member; every representative is in its own cluster.
The 30/80 rule therefore sits near K = 1,048,576 in granularity, four times finer than arm 1.

Submitted 76300: `run_kmeans.sbatch esmc6b raw 262144,65536,1048576` on standard_expand_2,
64 CPUs (the 43 GB reference matrix needs the 256 GB allocation). Quality job 76284 started
on node-64-768g-10.

## 2026-11-11 04:50 MST: first quality numbers (ESM2-8M grid vs MMseqs), a clear negative

Job 76284 (quality prep + precision done; TM-align still running). Recall = share of labelled
pairs whose genes share a token; precision = pair-uniform same-token cross-genome pairs with
SW E <= 1e-3 (200,000 draws each; 3,861,617 unique pairs aligned, 4 returned no alignment).

| arm | tokens | V | sd_vis | sd_invis | blk650_vis | blk650_invis | random | seqprec pair | seqprec token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmseqs id30c80 | 1,038,303 | 1 | **0.555** | 0.0052 | **0.722** | 0.0028 | 6e-5 | **0.994** | 0.980 |
| 8m raw K65536 | 65,550 | 0.795 | 0.086 | 0.0082 | 0.322 | 0.0168 | 2e-5 | 0.421 | 0.410 |
| 8m raw K262144 | 262,158 | 0.827 | 0.040 | 0.0026 | 0.186 | 0.0079 | 1e-5 | 0.490 | 0.474 |
| 8m raw K1048576 | 1,048,569 | 0.853 | 0.014 | 0.0005 | 0.078 | 0.0026 | 0 | 0.557 | 0.559 |
| 8m globalcentred K65536 | 65,531 | 0.798 | 0.094 | 0.0091 | 0.349 | 0.0178 | 3e-5 | 0.449 | 0.407 |
| 8m globalcentred K262144 | 262,164 | 0.829 | 0.043 | 0.0028 | 0.200 | 0.0088 | 1e-5 | 0.518 | 0.465 |
| 8m globalcentred K1048576 | 1,048,582 | 0.854 | 0.015 | 0.0006 | 0.083 | 0.0031 | 0 | 0.582 | 0.551 |
| 8m genomecentred K65536 | 65,539 | 0.806 | 0.094 | 0.0102 | 0.349 | 0.0143 | 1e-5 | 0.482 | 0.427 |
| 8m genomecentred K262144 | 262,158 | 0.833 | 0.044 | 0.0026 | 0.202 | 0.0072 | 0 | 0.537 | 0.476 |
| 8m genomecentred K1048576 | 1,048,601 | 0.855 | 0.015 | 0.0006 | 0.082 | 0.0026 | 0 | 0.587 | 0.562 |

n: sd_vis 3,955,569; sd_invis 403,491; blk650_vis 759,235; blk650_invis 293,345; random
300,000 (Wilson half-widths are below 0.001 for every recall here).

Reading against the predeclared criteria:
- **F2 holds for ESM2-8M**: recall on sequence-visible Spacedust pairs is 1.4-9.4% at every K
  and transform, against 55.5% for MMseqs. Embedding k-means tokens are a poor family stand-in
  on this model.
- F1 holds at K65536 and K262144 for raw (42%, 49%) and is narrowly missed at K1048576 (56-59%).
- F3 does not hold: collision rates are <= 3e-5.
- Genome-centring is the best transform at every K (precision +5 to +6 points over raw,
  recall equal or slightly higher), in line with the "lineage offset is noise" direction.
- V-measure (0.80-0.86) rewards fine partitions through homogeneity and says little here.

Diagnosis (login-node check, 300 pairs per set, ESM2-8M raw): labelled pairs have high
cosine (sd_vis median 0.966, blk650_vis 0.985, random 0.788), yet only 32% of sd_vis pairs and
59% of blk650_vis pairs share even a **coarse** cell (4,096 cells, about 1,000 genes each);
the mean cosine of a vector to its coarse centroid is 0.981. The quantization cells are
smaller than the spread of a family, so homologs straddle cell boundaries. This is a
property of hard equal-volume partitions of a continuum, and more K makes it worse.
A method with adaptive cluster extent is the natural test: the graph arm (step 5) was
started now on ESM2-8M genome-centred (job 76323, mutual 20-NN, IVF nprobe 32 over the
4,096 coarse centroids, Leiden modularity at resolutions 1, 10, 100, 1000).

Also submitted: 76324/76325 (ESM2-650M and ESM-C 600M genome-centred, all K) and quality job
76326 for the six raw ESM2-650M / ESM-C 600M arms (afterok 76276).

## 2026-11-11 05:08 MST: model arms, TM-align was I/O-bound, restructured

Tokens now in `$RD/tokens/` for ESM2-650M, ESM-C 600M and ESM-C 6B, raw and genome-centred,
each at K65536 / 262144 / 1048576 (jobs 76276, 76277, 76300 (6B, 64 CPU), 76324, 76325,
76345 (6B, 64 CPU)). ESM2-650M assigns 4,181,264 genes (75,331 reference + 936 query proteins
> 1,022 aa have no vector); ESM-C models assign all 4,257,531. Global-centring was not run for
the large models: on ESM2-8M it sat between raw and genome-centred at every K, so raw +
genome-centred at three K covers "raw K262144 plus the best two settings" of the brief.

Prep/precision for the six raw ESM2-650M / ESM-C 600M arms (job 76326, before cancellation):

| arm | sd_vis | sd_invis | blk650_vis | blk650_invis | seqprec pair | seqprec token |
|---|---:|---:|---:|---:|---:|---:|
| 650m raw K65536 | 0.124 | 0.0081 | 0.513 | 0.0466 | 0.456 | 0.528 |
| 650m raw K262144 | 0.056 | 0.0025 | 0.292 | 0.0249 | 0.559 | 0.622 |
| 650m raw K1048576 | 0.018 | 0.0006 | 0.111 | 0.0074 | 0.669 | 0.713 |
| esmc600m raw K65536 | 0.068 | 0.0021 | 0.280 | 0.0148 | 0.199 | 0.247 |
| esmc600m raw K262144 | 0.038 | 0.0007 | 0.190 | 0.0087 | 0.320 | 0.377 |
| esmc600m raw K1048576 | 0.014 | 0.0003 | 0.082 | 0.0034 | 0.466 | 0.525 |

(blk650 pairs were selected by ESM2-650M cosine, so the 650M arms are favoured on that set.)
F2 holds for both models: sd_vis recall <= 12.4% everywhere. ESM-C 600M is the worst model
on precision (20-47%), consistent with finding G of the brief (ESM-C lineage signal).

**Correction / operational finding.** The TM-align step of quality jobs 76284 and 76326 was
I/O-bound: `foldseek tmalign` used 1-4 of 24 cores (about 38,000 of 174,866 pairs in 33 min)
because it read the 9.7 GB AFDB structure database at random from network storage. Both jobs
were cancelled during TM (their prep and precision outputs are complete and kept), as were the
queued quality jobs 76346, 76348, 76357. `run_quality.sbatch` now reads the six foldseek data
files sequentially into page cache first and passes `--db-load-mode 3`; a fourth argument
SKIP_PREP=1 runs TM only. Resubmitted: 76361 (`tm_a`, TM only, the 16 arms with precision done)
and 76362 (`models_rest`, full quality with TM for the 12 remaining k-means arms). The same
I/O limit probably slows the coordinator's TM jobs (76209, 76134) on the same database.

## 2026-11-11 05:20 MST: TM-align parallelism, recall by identity, hybrid arm started

**Correction 2.** Page-cache warming alone did not fix TM-align (job 76361, cancelled after
3.8 min): `foldseek tmalign` parallelizes over the targets of one query entry, and our fake
prefilter has 1-4 targets per query, so 1-4 threads ran. `run_quality.sbatch` now splits the
pairs into $SLURM_CPUS_ON_NODE chunks and runs one single-threaded foldseek per chunk
(smoke-tested on 6 pairs). Job 76362 also failed in `precision` on a degenerate MMseqs row
(`seqid` printed as `0.-2147483648`, qstart -1, E 9.9e-324); `quality.py precision` now reads
all fields as text and treats rows with qstart < 0 or an unparsable seqid/E as "no alignment"
(1 such row in each of grid8m and models_raw; the recomputed precisions are identical to 5
decimals). Resubmitted: 76363 (`run_reprec_tm.sbatch`: precision recomputed for the 16 arms,
then chunked TM for them) and 76364 (`models_rest`, full quality with TM for 12 arms).

**Remaining k-means arms (prep + precision from 76364).**

| arm | copies/genome | sd_vis | sd_invis | blk650_vis | blk650_invis | seqprec pair | seqprec token |
|---|---:|---:|---:|---:|---:|---:|---:|
| 650m genomecentred K65536 | 1.348 | **0.159** | 0.0140 | 0.605 | 0.0495 | 0.582 | 0.596 |
| 650m genomecentred K262144 | 1.196 | 0.064 | 0.0034 | 0.333 | 0.0254 | 0.639 | 0.673 |
| 650m genomecentred K1048576 | 1.102 | 0.020 | 0.0007 | 0.122 | 0.0066 | **0.694** | 0.751 |
| esmc600m genomecentred K65536 | 1.382 | 0.084 | 0.0032 | 0.318 | 0.0150 | 0.276 | 0.317 |
| esmc600m genomecentred K262144 | 1.198 | 0.043 | 0.0014 | 0.207 | 0.0092 | 0.366 | 0.431 |
| esmc600m genomecentred K1048576 | 1.100 | 0.015 | 0.0004 | 0.086 | 0.0035 | 0.472 | 0.567 |
| esmc6b raw K65536 | **2.555** | 0.049 | 0.0017 | 0.211 | 0.0158 | 0.165 | 0.203 |
| esmc6b raw K262144 | 1.629 | 0.029 | 0.0006 | 0.148 | 0.0092 | 0.257 | 0.324 |
| esmc6b raw K1048576 | 1.218 | 0.011 | 0.0002 | 0.065 | 0.0034 | 0.391 | 0.483 |
| esmc6b genomecentred K65536 | 1.599 | 0.068 | 0.0044 | 0.247 | 0.0167 | 0.262 | 0.294 |
| esmc6b genomecentred K262144 | 1.305 | 0.034 | 0.0014 | 0.163 | 0.0097 | 0.321 | 0.397 |
| esmc6b genomecentred K1048576 | 1.140 | 0.012 | 0.0004 | 0.068 | 0.0038 | 0.401 | 0.532 |

- ESM-C 6B raw tokens hold 2.56 copies per carrying genome at K65536 (ESM2-8M raw 1.32,
  MMseqs 1.08): its cells collect proteins of one genome, the lineage signal of finding G.
  Genome-centring lowers this to 1.60 and raises precision 16.5% -> 26.2%.
- Genome-centring raises precision for every model (650M: +12.6, +8.1, +2.5 points at
  K65536 / 262144 / 1048576).
- Best embedding arm on every axis: ESM2-650M genome-centred. It still fails F2 (15.9%).

**Recall by pairwise identity** (`recall_strata.py`, job 76374 after a column-name fix in
76370; `$RD/ws_tokens/recall_strata.tsv`), sequence-visible native Spacedust pairs:

| identity | n | MMseqs | 650m raw K65536 | 8m genomecentred K65536 | 6b genomecentred K65536 |
|---|---:|---:|---:|---:|---:|
| < 30% | 745,389 | 0.130 | 0.011 | 0.015 | 0.007 |
| 30-50% | 2,094,306 | 0.568 | 0.064 | 0.053 | 0.025 |
| 50-70% | 932,124 | 0.801 | 0.262 | 0.179 | 0.141 |
| 70-90% | 177,384 | 0.883 | 0.556 | 0.451 | 0.431 |
| >= 90% | 6,366 | 0.931 | 0.779 | 0.726 | 0.715 |

At >= 90% identity and >= 80% coverage (6,009 pairs) MMseqs shares a token for 97.3% and the
best k-means arm for 81.4%. By target stratum, recall on other-phylum pairs (3,142,700) is
45.0% for MMseqs and 0.2-1.8% for the k-means arms. The embedding tokens are worse than the
sequence clusters in every identity bin, including near-identical pairs.

**Hybrid arm started** (job 76375, `hybrid_tokens.py --model esm2_650m --transform
genomecentred`): MMseqs families merged when their ESM2-650M genome-centred centroids are
mutual 10-NN with cosine above the (1 - p) quantile of 2,000,000 random family-centroid
pairs, p in {1e-3, 1e-4, 1e-5}. Question: can embeddings add remote homology on top of
sequence families? Predeclared: it counts as useful if recall on sequence-invisible TM >= 0.5
pairs (or, until tm_key lands, on sd_invis and blk650_invis) rises by >= 0.05 absolute over
MMseqs while sequence precision stays >= 0.9 and homology precision >= 0.95.

## 2026-11-11 05:40 MST: structural precision of same-token pairs (TM-align done for 28 arms)

Jobs 76363 (16 arms) and 76364 (12 arms), chunked single-thread foldseek TM-align, about
0.05 s per pair per core (291,449 + 8,790 + 688 pairs in 27 min; 232,405 + 6,836 + 705 in
21 min). Per arm: up to 20,000 sequence-invisible pairs from the pair-uniform sample with both
AFDB models; TM = max(qTM, tTM) (normalized by the shorter chain); homology precision =
seqprec + (1 - seqprec) x TMconf.

| arm | seqprec pair | TMconf (invisible) | homology prec |
|---|---:|---:|---:|
| mmseqs id30c80 | 0.994 | 0.949 (n = 1,032) | **0.9997** |
| 8m raw K65536 / K262144 / K1048576 | 0.421 / 0.490 / 0.557 | 0.314 / 0.289 / 0.267 | 0.603 / 0.637 / 0.675 |
| 8m globalcentred K65536 / K262144 / K1048576 | 0.449 / 0.518 / 0.582 | 0.339 / 0.313 / 0.286 | 0.636 / 0.669 / 0.702 |
| 8m genomecentred K65536 / K262144 / K1048576 | 0.482 / 0.537 / 0.587 | 0.342 / 0.312 / 0.290 | 0.659 / 0.681 / 0.707 |
| 650m raw K65536 / K262144 / K1048576 | 0.456 / 0.559 / 0.669 | 0.238 / 0.252 / 0.280 | 0.585 / 0.670 / 0.762 |
| 650m genomecentred K65536 / K262144 / K1048576 | 0.582 / 0.639 / 0.694 | 0.311 / 0.268 / 0.256 | 0.712 / 0.736 / **0.773** |
| esmc600m raw K65536 / K262144 / K1048576 | 0.199 / 0.320 / 0.466 | 0.101 / 0.122 / 0.141 | 0.280 / 0.403 / 0.541 |
| esmc600m genomecentred K65536 / K262144 / K1048576 | 0.276 / 0.366 / 0.472 | 0.131 / 0.143 / 0.148 | 0.371 / 0.457 / 0.551 |
| esmc6b raw K65536 / K262144 / K1048576 | 0.165 / 0.257 / 0.391 | 0.099 / 0.115 / 0.133 | 0.247 / 0.343 / 0.472 |
| esmc6b genomecentred K65536 / K262144 / K1048576 | 0.262 / 0.321 / 0.401 | 0.150 / 0.149 / 0.150 | 0.373 / 0.422 / 0.491 |

n for TMconf is 20,011-20,068 per k-means arm (Wilson half-width <= 0.0065).

- Sequence-invisible same-token pairs of the embedding arms are mostly **not** shared folds:
  10-34% reach TM >= 0.5. The embedding tokens add few remote homologs per false pair.
- No k-means arm reaches the predeclared 80% homology-precision gate (best 77.3%, ESM2-650M
  genome-centred K1048576, whose sd_vis recall is 2.0%). Under the predeclared rule the
  engine gets the MMseqs arm alone from the k-means family.
- B1-B4 ("better than sequence clusters") cannot be met: B2 needs homology precision
  >= 0.9 x 0.9997.

## 2026-11-11 05:50 MST: graph and hybrid arms written; TM label sets available

**Graph arm** (job 76323, 72 min, 24 CPU): ESM2-8M genome-centred, 20-NN by faiss IVFFlat
(4,096 lists = the coarse k-means centroids, nprobe 32; the search took 3,948 s, memory-
bandwidth bound), 18,708,674 mutual edges, 172,153 genes with no mutual edge (singleton
tokens), mean degree 8.79, edge cosine 0.797 / 0.933 / 0.992 (1% / 50% / 99%). Leiden
(modularity) at resolution 1, 10, 100, 1000 gave 192,882 / 193,971 / 197,873 / (see INDEX)
communities: resolution barely moves the partition, because most structure is set by the
disconnected components of the mutual graph.

**Hybrid arm** (job 76376 after an import error in 76375: `graph_tokens` needs the venv;
29 min, peak RSS 42.9 GB): MMseqs families with an ESM2-650M genome-centred centroid
(1,016,005 of 1,038,303 families). Random family-centroid cosine: median 0.213, 99th 0.832,
99.9th 0.908, 99.99th 0.943, 99.999th 0.959 (2,000,000 pairs). Mutual 10-NN family pairs:
1,739,324 (cosine median 0.869). Merged token counts: 730,345 (p = 1e-3, tau 0.908),
882,216 (p = 1e-4, tau 0.943), 956,213 (p = 1e-5, tau 0.959).

**Correction 3 (sampler).** The first graph quality job (76386) stopped on an assertion in the
token-uniform sampler: with giant communities the float64 cumulative weights reach about
1e13, and `lo + rand x (hi - lo)` rounded up into the next token. The sampler now draws
integers from integer cumulative weights (`rng.integers`), which is exact; re-tested on the
synthetic case and on a 2,000,000-gene token. The k-means samples drawn earlier passed the
assertion and stand (their weight totals are below 1e10); they were produced by code
snapshot 759ab600... / bf97f9a5..., the graph and hybrid samples by the integer version.

`$S/work_key/tm_key.parquet` landed at 04:45 PST (job 76209). Submitted: 76388 (labels with
tm_hi/tm_lo), 76389 (hybrid quality, afterok 76388), 76387 (graph quality), 76390
(`run_finalize.sbatch`: recall for every arm on the new label sets, `quality.tsv`,
`recall_strata.tsv`; afterok 76388, afterany 76387 76389).

## 2026-11-11 06:00 MST: graph and union-find hybrid quality: adaptive extent raises recall, chaining destroys precision

Quality jobs 76387 (graph) and 76389 (hybrid, union-find). tm_hi n = 72,372, tm_lo n =
497,092 (sequence-invisible in-block pairs of any model with both structures, split at
TM 0.5; 12.7% of the 569,464 are TM >= 0.5). Recall columns below are from prep, which ran on
the new label file for these two jobs.

| arm | tokens | max size | sd_vis | sd_invis | tm_hi | tm_lo | random | seqprec pair | seqprec token | TMconf | homology prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| graph 8m gnm r1 | 192,882 | 479,943 | 0.846 | 0.667 | 0.669 | 0.120 | 0.0225 | 0.017 | 0.522 | 0.066 | 0.082 |
| graph 8m gnm r10 | 193,971 | 70,757 | 0.803 | 0.528 | 0.537 | 0.050 | 0.0027 | 0.079 | 0.526 | 0.179 | 0.244 |
| graph 8m gnm r100 | 197,873 | 7,394 | 0.556 | 0.264 | 0.372 | 0.026 | 0.00032 | 0.337 | 0.529 | 0.375 | 0.586 |
| graph 8m gnm r1000 | 218,156 | 868 | 0.227 | 0.046 | 0.217 | 0.014 | 0.00005 | 0.571 | 0.504 | 0.451 | 0.764 |
| hybrid 650m gnm cc p=1e-3 | 730,345 | 266,762 | 0.734 | 0.057 | 0.215 | 0.035 | 0.0038 | 0.038 | 0.893 | 0.087 | 0.122 |
| hybrid 650m gnm cc p=1e-4 | 882,216 | 154,191 | 0.666 | 0.014 | 0.058 | 0.016 | 0.0013 | 0.072 | 0.948 | 0.077 | 0.143 |
| hybrid 650m gnm cc p=1e-5 | 956,213 | 91,994 | 0.638 | 0.009 | 0.030 | 0.007 | 0.0006 | 0.163 | 0.967 | 0.072 | 0.223 |
| mmseqs id30c80 | 1,038,303 | 5,362 | 0.555 | 0.005 | (finalize) | (finalize) | 0.00006 | 0.994 | 0.980 | 0.949 | 0.9997 |

- **Graph communities beat k-means cells at equal precision.** Graph r1000 against k-means
  ESM2-8M genome-centred K262144: sequence precision 0.571 vs 0.537, sd_vis recall 0.227 vs
  0.044, sd_invis 0.046 vs 0.003. Graph r100 matches MMseqs on sd_vis (0.556 vs 0.555) and
  reaches 0.264 on sd_invis and 0.372 on tm_hi, with tm_hi / tm_lo recall 14x apart (fold
  discrimination, B3 satisfied). Its precision is the problem: 0.337 sequence, 0.586 homology.
- Leiden resolution (modularity) controls the giant communities; the community count barely
  moves because 172,153 genes have no mutual edge and stay singletons.
- **Union-find merging of MMseqs families percolates.** Even at p = 1e-5 (tau 0.959) the
  largest merged token holds 91,994 genes; pair-uniform precision falls to 4-16% because a
  few giant tokens dominate the pairs, while the typical token stays clean (token-uniform
  precision 0.89-0.97). Collision rates 0.06-0.38% fail B4. Negative result for the chaining
  merge. A non-chaining merge (Leiden CPM at resolution tau on the same family graph) is
  running as job 76391.

## 2026-11-11 06:30 MST: CPM hybrid passes the gates; graph precision plateaus at 0.79

**Graph, higher resolutions** (job 76394 from the cached kNN, Leiden only; quality 76396):

| graph 8m gnm | tokens | max size | sd_vis | sd_invis | tm_hi | tm_lo | random | seqprec pair | TMconf | homology prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r1000 | 218,156 | 868 | 0.227 | 0.046 | 0.217 | 0.014 | 5e-5 | 0.571 | 0.451 | 0.764 |
| r3000 | 247,268 | 400 | 0.169 | 0.021 | 0.139 | 0.010 | 2e-5 | 0.627 | 0.437 | **0.790** |
| r10000 | 320,228 | 136 | 0.092 | 0.008 | 0.074 | 0.006 | 1e-5 | 0.656 | 0.400 | 0.794 |
| r30000 | 468,750 | 66 | 0.042 | 0.003 | 0.034 | 0.003 | 1e-5 | 0.659 | 0.379 | 0.788 |

Homology precision of the ESM2-8M graph saturates at 0.79: one third of the same-community
pairs are neither sequence-visible nor same-fold however small the communities get, so the
residual error sits in the mutual-kNN edges themselves. It never reaches the 0.80 gate.

**Hybrid with Leiden CPM merge** (job 76391, 31.8 min; quality 76400). Same family graph as
the union-find arms, partitioned by CPM at resolution tau_p:

| arm | tokens | max size | sd_vis | sd_invis | blk650_vis | tm_hi | tm_lo | random | seqprec pair | seqprec token | homology prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mmseqs id30c80 | 1,038,303 | 5,362 | 0.555 | 0.0052 | 0.722 | 0.0106 | 0.0004 | 6e-5 | 0.994 | 0.980 | 0.9997 |
| hybrid cpm p=1e-3 (tau 0.908) | 874,916 | 10,067 | **0.640** | 0.0123 | 0.812 | **0.0438** | 0.0019 | 1.1e-4 | 0.987 | 0.699 | 0.9992 |
| hybrid cpm p=1e-4 (tau 0.943) | 956,330 | 9,955 | 0.619 | 0.0081 | 0.790 | 0.0251 | 0.0011 | 1.0e-4 | 0.991 | 0.815 | 0.9994 |
| hybrid cpm p=1e-5 (tau 0.959) | 995,516 | 9,774 | 0.606 | 0.0069 | 0.772 | 0.0184 | 0.0008 | 1.0e-4 | 0.991 | 0.889 | 0.9994 |

95% Wilson intervals: tm_hi recall 0.0424-0.0454 (cpm 1e-3) vs 0.0099-0.0114 (MMseqs);
pair-uniform sequence precision 0.9866-0.9876 vs 0.9935-0.9942.

- CPM removes the percolation: largest token 10,067 genes (union-find: 266,762).
- The hybrid adds 8.5 points of recall on sequence-visible Spacedust pairs (0.555 -> 0.640)
  and 9.0 points on blk650_vis at pair precision 0.987: embedding centroids rejoin
  sequence families that MMseqs' cascaded set-cover clustering split.
- Remote homology gain is small: tm_hi recall 1.1% -> 4.4% (+3.3 points, short of the
  predeclared +5 points), sd_invis 0.5% -> 1.2%.
- Token-uniform precision drops to 0.70 at p = 1e-3: the merged small families are often
  not sequence-visible to each other; pair-uniform precision is dominated by the large,
  intact families.
- Under the predeclared recommendation rule this is the only arm with embedding input that
  clears homology precision >= 0.80 and collision <= 0.1%.

Running: 76404 (CPM hybrid at p = 1e-2, 3e-2, 1e-1 from the cached family graph) to map how
far the threshold can drop before precision fails; 76395 (ESM2-650M genome-centred graph,
64 CPUs).

## 2026-11-11 06:45 MST: CPM hybrid threshold sweep

Job 76404 (CPM hybrid from the cached family graph, 51 s; the manifests' `seconds` for these
cached arms exclude the 1,867 s family-graph build of job 76391), quality 76405.
tau = 0.832 (p = 1e-2), 0.764 (3e-2), 0.630 (1e-1).

| hybrid cpm | tokens | max | sd_vis | sd_invis | blk650_vis | tm_hi (95% CI) | tm_lo | random | seqprec pair | seqprec token | TMconf invis (n) | homology prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p=1e-1 | 608,932 | 11,918 | 0.704 | 0.044 | 0.855 | 0.093 (0.091-0.095) | 0.0038 | 1.4e-4 | 0.945 | 0.449 | 0.957 (9,572) | 0.9977 |
| p=3e-2 | 655,657 | 10,081 | 0.683 | 0.036 | 0.845 | 0.083 (0.081-0.085) | 0.0033 | 1.1e-4 | 0.962 | 0.487 | 0.962 (6,576) | 0.9985 |
| p=1e-2 | 727,649 | 10,067 | 0.672 | 0.026 | 0.838 | 0.075 (0.073-0.077) | 0.0029 | 1.0e-4 | 0.974 | 0.547 | 0.956 (4,451) | 0.9989 |
| p=1e-3 | 874,916 | 10,067 | 0.640 | 0.012 | 0.812 | 0.044 (0.042-0.045) | 0.0019 | 1.1e-4 | 0.987 | 0.699 | 0.938 (2,183) | 0.9992 |
| MMseqs | 1,038,303 | 5,362 | 0.555 | 0.005 | 0.722 | 0.011 (0.010-0.011) | 0.0004 | 6e-5 | 0.994 | 0.980 | 0.949 (1,032) | 0.9997 |

- The sequence-invisible pairs that the CPM hybrid adds are mostly shared folds
  (TMconf 0.94-0.96 among pair-uniform invisible pairs), unlike the k-means and graph arms
  (0.10-0.45).
- At p <= 1e-2 the hybrid meets the predeclared hybrid criterion: tm_hi recall +6.5 to +8.2
  points over MMseqs, sequence precision >= 0.945, homology precision >= 0.997.
- Token-uniform sequence precision falls to 0.45-0.55: many small merged tokens join
  families with no pairwise SW hit. Their structural status is the open question for the
  engine (it scores tokens, and most tokens are small); a token-uniform TM check is queued.

Queued: 76413 (CPM hybrid p = 0.3, 0.5) -> 76414 (quality) -> 76420 (TM-align of 20,000
token-uniform invisible pairs per arm for MMseqs, the eight CPM hybrids, graph r100 / r3000
and three k-means arms; `quality.py --tm-sampler token`, new stats keys `tm_conf_token`,
`homology_prec_token`).

## 2026-11-11 06:55 MST: CPM hybrid at lower resolution: the precision gate still does not bind

Job 76413 (p = 0.3, tau 0.392; p = 0.5, tau 0.213, the null median) and 76443 (CPM at explicit
resolution 0.1 / 0.05 / 0.02 on all 1,739,324 mutual family edges); quality 76414, 76444.

| hybrid cpm | tokens | sd_vis | sd_invis | blk650_vis | tm_hi | tm_lo | random | seqprec pair | seqprec token | TMconf invis | homology prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p=0.3 | 529,526 | 0.722 | 0.056 | 0.870 | 0.127 | 0.0054 | 1.5e-4 | 0.923 | 0.403 | 0.925 | 0.9942 |
| p=0.5 | 453,511 | 0.752 | 0.080 | 0.886 | 0.167 | 0.0072 | 1.7e-4 | 0.884 | 0.376 | 0.896 | 0.9879 |
| r=0.1 | 380,802 | 0.776 | 0.118 | 0.902 | 0.220 | 0.0096 | 1.9e-4 | 0.851 | 0.362 | 0.879 | 0.9819 |
| r=0.05 | 334,660 | 0.799 | 0.142 | 0.917 | 0.259 | 0.0124 | 2.0e-4 | 0.811 | 0.358 | 0.819 | 0.9658 |
| r=0.02 | 295,182 | 0.828 | 0.185 | 0.932 | 0.315 | 0.0161 | 2.3e-4 | 0.752 | 0.368 | 0.779 | 0.9450 |

- The CPM hybrid dominates the graph arm: at r = 0.02 it holds sd_vis 0.828 (graph r100
  0.556), sd_invis 0.185 (0.264), tm_hi 0.315 (0.372) at homology precision 0.945 (0.586).
- tm_hi / tm_lo recall stays 19-24x apart along the sweep (fold discrimination).
- Collision stays at 1.5-2.3e-4, below the 1e-3 gate.
- Bias to keep in mind: tm_hi / tm_lo and blk650 pairs were chosen as ESM anchor pairs, so
  arms built on ESM cosine are favoured on them; sd_invis (native Spacedust, profile-based)
  carries no such selection and moves the same way (0.005 -> 0.185).
- The predeclared selection rule (highest tm_hi among arms with homology precision >= 0.80
  and collision <= 0.1%) keeps preferring the loosest resolution tested. Job 76448 maps
  r = 0.01 / 0.005 / 0.002 to find where a gate binds (quality 76449); 76450 adds the
  token-uniform TM check for all six r-arms.

## 2026-11-11 07:05 MST: token-uniform TM check: small merged tokens are mostly not shared folds

Job 76420 (`--tm-sampler token`): TM-align of sequence-invisible pairs from the
token-uniform sample (every token with a cross-genome pair equally likely).

| arm | seqprec token | TMconf token (n) | homology prec token | homology prec pair |
|---|---:|---:|---:|---:|
| mmseqs id30c80 | 0.980 | 0.822 (2,822) | **0.9965** | 0.9997 |
| hybrid cpm p=1e-5 | 0.889 | 0.241 | 0.916 | 0.9994 |
| hybrid cpm p=1e-4 | 0.815 | 0.196 | 0.851 | 0.9994 |
| hybrid cpm p=1e-3 | 0.699 | 0.219 | 0.765 | 0.9992 |
| hybrid cpm p=1e-2 | 0.547 | 0.288 | 0.677 | 0.9989 |
| hybrid cpm p=3e-2 | 0.487 | 0.321 | 0.652 | 0.9985 |
| hybrid cpm p=1e-1 | 0.449 | 0.346 | 0.640 | 0.9977 |
| hybrid cpm p=0.3 | 0.403 | 0.347 | 0.610 | 0.9942 |
| hybrid cpm p=0.5 | 0.376 | 0.350 | 0.595 | 0.9879 |
| graph 8m gnm r100 | 0.529 | 0.369 | 0.703 | 0.586 |
| graph 8m gnm r3000 | 0.495 | 0.294 | 0.644 | 0.790 |
| kmeans 650m gnm K65536 | 0.596 | 0.391 | 0.754 | 0.712 |
| kmeans 650m gnm K1048576 | 0.751 | 0.398 | 0.850 | 0.773 |
| kmeans 8m gnm K262144 | 0.476 | 0.324 | 0.646 | 0.681 |

**This changes the reading of the hybrid.** Pair-uniform precision is carried by the large
MMseqs families, which the merge leaves intact; the typical token that the merge creates
joins small families (often singletons) whose invisible pairs reach TM >= 0.5 only 20-35% of
the time. Per token, the CPM hybrids at p >= 1e-3 are no purer than the k-means or graph
arms. Only p = 1e-5 and 1e-4 stay above 0.85 per token, and they add little recall
(tm_hi 0.018 / 0.025 against MMseqs 0.011).

**Correction 4 (TM scoring subset).** `tmcollect` scored each arm on every invisible pair of
that arm that had a TM-score in the batch, including pairs sampled for other arms of the
batch. When an arm had more than 20,000 candidates, pairs drawn for sibling arms entered its
estimate (n 20,011-20,068 for k-means arms; 13,133-38,072 in the token batch above, where the
hybrids share many pairs). `tmprep` now records each arm's own requests and `tmcollect`
scores only those. Job 76453 (`run_recollect_tm.sbatch`) re-scores every batch from the
TM-align outputs on disk (no realignment) and rebuilds `quality.tsv`; the numbers in this
entry and the TM columns of earlier entries will be superseded by its output.

## 2026-11-11 07:25 MST: final summary, ablation table and recommendation

(The harness refuses a REPORT.md from this subagent session; this entry is the report. All
TM columns below are the re-scored values of job 76460, own-sample only.)

### What was tested

52 token arms over all 4,257,531 genes, each in `$RD/tokens/<arm>.parquet` + `.json`, one
line each in `$RD/tokens/INDEX.tsv`:
- `mmseqs__seq__id30c80` (sequence families, the baseline and reference labelling);
- 30 k-means arms: ESM2-8M {raw, globalcentred, genomecentred}, ESM2-650M / ESM-C 600M /
  ESM-C 6B {raw, genomecentred}, each at K = 65,536 / 262,144 / 1,048,576;
- 7 graph arms: ESM2-8M genome-centred mutual 20-NN graph, Leiden modularity at resolution
  1 to 30,000 (ESM2-650M graph: job 76395 still running at this entry; its quality job 76462
  and a final `run_finalize` 76463 are queued with dependencies and will add 4 rows to
  `quality.tsv`);
- 17 hybrid arms: MMseqs families merged by ESM2-650M genome-centred centroid similarity,
  union-find (3, percolate) or Leiden CPM (14, sweep of the merge stringency).

### Ablation table (representative rows; every arm is in `quality.tsv`)

Recall columns: share of labelled pairs sharing a token. Precision: same-token
cross-genome pairs, 200,000 per sampler; "hom" = sequence-visible or TM >= 0.5.

| arm | tokens | sd_vis | sd_invis | tm_hi | tm_lo | random | seq prec (pair) | hom prec (pair) | hom prec (token) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **mmseqs id30c80** | 1,038,303 | 0.555 | 0.005 | 0.011 | 0.0004 | 6e-5 | 0.994 | 0.9997 | 0.997 |
| kmeans 8m raw K262144 (arm 1) | 262,158 | 0.040 | 0.003 | 0.030 | 0.003 | 1e-5 | 0.490 | 0.637 | |
| kmeans 8m gnm K65536 | 65,539 | 0.094 | 0.010 | 0.062 | 0.004 | 1e-5 | 0.482 | 0.659 | |
| kmeans 8m gnm K262144 | 262,158 | 0.044 | 0.003 | 0.027 | 0.002 | 0 | 0.537 | 0.681 | 0.645 |
| kmeans 650m gnm K65536 | 65,494 | 0.159 | 0.014 | 0.149 | 0.012 | 0 | 0.582 | 0.712 | 0.753 |
| kmeans 650m gnm K1048576 | 1,048,563 | 0.020 | 0.001 | 0.016 | 0.002 | 0 | 0.694 | 0.772 | 0.850 |
| kmeans esmc600m gnm K65536 | 65,536 | 0.084 | 0.003 | 0.053 | 0.006 | 3e-5 | 0.276 | 0.371 | |
| kmeans esmc6b raw K65536 | 65,547 | 0.049 | 0.002 | 0.054 | 0.007 | 1e-5 | 0.165 | 0.247 | |
| kmeans esmc6b gnm K1048576 | 1,048,555 | 0.012 | 0.000 | 0.010 | 0.001 | 0 | 0.401 | 0.490 | |
| graph 8m gnm r100 | 197,873 | 0.556 | 0.264 | 0.372 | 0.026 | 3e-4 | 0.337 | 0.586 | 0.703 |
| graph 8m gnm r3000 | 247,268 | 0.169 | 0.021 | 0.139 | 0.010 | 2e-5 | 0.627 | 0.790 | 0.642 |
| hybrid union-find p=1e-5 | 956,213 | 0.638 | 0.009 | 0.030 | 0.007 | 6e-4 | 0.163 | 0.223 | |
| **hybrid cpm p=1e-4** | 956,330 | 0.619 | 0.008 | 0.025 | 0.001 | 1e-4 | 0.991 | 0.9994 | 0.852 |
| hybrid cpm p=1e-3 | 874,916 | 0.640 | 0.012 | 0.044 | 0.002 | 1e-4 | 0.987 | 0.9992 | 0.772 |
| hybrid cpm p=1e-1 | 608,932 | 0.704 | 0.044 | 0.093 | 0.004 | 1e-4 | 0.945 | 0.9976 | 0.680 |
| **hybrid cpm r=0.05** | 334,660 | 0.799 | 0.142 | 0.259 | 0.012 | 2e-4 | 0.811 | 0.966 | 0.601 |
| hybrid cpm r=0.002 | 254,027 | 0.898 | 0.417 | 0.472 | 0.028 | 4e-4 | 0.542 | 0.848 | 0.640 |

n: sd_vis 3,955,569; sd_invis 403,491; tm_hi 72,372; tm_lo 497,092; random 300,000.

Recall on sequence-visible Spacedust pairs by target stratum (other phylum, n = 3,142,700):
MMseqs 0.450, hybrid cpm p=1e-4 0.508, p=1e-1 0.595, r=0.05 0.698, r=0.002 0.830,
graph r100 0.447, kmeans 650m gnm K65536 0.029. By identity < 30% (n = 745,389): 0.130,
0.168, 0.255, 0.447, 0.734, 0.353, 0.018.

### Predeclared criteria

- F1 (pair sequence precision < 50% at every K): holds for ESM-C 600M and 6B (all arms
  < 0.48), not for ESM2 (0.56-0.69 at K1048576).
- **F2 (sd_vis recall < 50% at every K): holds for every k-means arm (max 0.159).** k-means
  embedding tokens are a poor family stand-in on this collection.
- F3 (collision > 0.5%): holds for no arm except graph r1 (2.2%).
- B1-B4 (embedding tokens better than sequence clusters): not met by any k-means or graph
  arm (homology precision <= 0.79 against 0.9997).
- Hybrid criterion (tm_hi +0.05 over MMseqs, seq precision >= 0.9, homology >= 0.95): met by
  CPM p = 1e-2, 3e-2, 1e-1 (tm_hi 0.075-0.093); r = 0.1 and r = 0.05 meet the recall and
  homology parts with sequence precision 0.85 / 0.81.
- Recommendation rule (highest tm_hi among arms with homology precision >= 0.80 and
  collision <= 0.1%): the gate never binds inside the CPM sweep, so the literal rule picks
  the loosest arm tested (r = 0.002). The rule was written for discrete arms and has no
  stopping point on a continuous stringency parameter; I do not use its literal output.

### Recommendation for the engine

1. **Primary: `mmseqs__seq__id30c80`.** 99.97% pair and 99.65% per-token homology precision,
   collision 6e-5. Its weakness is family completeness: 55.5% of visible Spacedust pairs, 45%
   of cross-phylum ones, 13% below 30% identity.
2. **Second arm for distant recurrence: `hybrid__esm2_650m__genomecentred__mm30cpmr0p05`.**
   Visible-pair recall 0.799 (cross-phylum 0.698), invisible 0.142, tm_hi 0.259 with tm_lo
   0.012 (fold discrimination 21x), pair homology precision 0.966, collision 2e-4. Its cost
   is per-token purity: 60% homology precision for a uniformly chosen token, against 99.7%
   for MMseqs. Recurrence calls found only under this arm should be checked for token purity
   (per-token TM) before they are believed.
3. **Conservative alternative: `...mm30cpm1em4`** (per-token homology 0.85, pair 0.9994):
   +6.4 points of visible recall over MMseqs, little remote homology.
4. Do not use k-means arms (including arm 1) as family tokens; they serve only as a
   negative control. If the engine needs an embedding-only tokenization for an ablation, use
   graph r100 (recall equal to MMseqs on visible pairs, homology precision 0.59).

### Costs (wall time, CPUs, peak RSS)

MMseqs clustering 692 s on 64 CPUs, 6.5 GB. k-means: ESM2-8M 147-181 s, ESM2-650M 513-735 s,
ESM-C 600M 451-611 s on 24 CPUs (10-41 GB); ESM-C 6B 436-894 s on 64 CPUs (81 GB); the
coarse codebook and data load are 80-90% of each. Graph ESM2-8M: 3,948 s IVF kNN (memory-
bandwidth bound) + 1-3 min per Leiden resolution, 18 GB. Hybrid: 1,867 s once (centroids +
family kNN), then 10-50 s per merge setting from the cache, 41 GB. Pair precision: MMseqs
SW of ~4 M pairs in 1.5 min; TM-align about 0.05 s per pair per core once chunked.

### Artifacts

- Tokens: `$RD/tokens/*.parquet`, `*.json`, `INDEX.tsv` (52 arms of this workstream; the
  `dev__esm2_8m__raw__K16384` files there are not from this workstream).
- Quality: `$RD/ws_tokens/quality.tsv` (one row per arm), `$RD/ws_tokens/recall_strata.tsv`,
  per-arm `$RD/ws_tokens/quality/<arm>/{stats.json, recall.tsv, sample.parquet,
  sample_aln.parquet, tm_invisible*.parquet}`; labels `$RD/ws_tokens/labels/`.
- Reusable intermediates: gid-keyed MMseqs DB `$RD/ws_tokens/work/mmseqs/seqdb` (+ FASTA
  `all.faa`, headers = gid); key-headed foldseek aliases `$RD/ws_tokens/work/fsdb/`; coarse
  codebooks `$RD/ws_tokens/work/kmeans/<model>__<transform>/`; ESM2-8M genome-centred 20-NN
  lists `$RD/ws_tokens/work/graph/esm2_8m__genomecentred/knn_k20_np32.npz`; hybrid family
  graph `$RD/ws_tokens/work/hybrid/esm2_650m__genomecentred/edges_k10_np32_*.npz`.
- Code: `$RD/ws_tokens/code/` (mirror `code/` here), snapshots by hash in
  `$RD/ws_tokens/code_snapshots/`; small results mirrored in `results/`.

### Open questions

- Per-token purity of the CPM hybrids: is the 60% driven by a few merge-prone fold classes
  (repeats, transporters), so that a per-token purity filter could rescue most merges?
- Whether a learned metric (ws_metric) or ESM2-650M graph communities (job 76395) lift the
  graph arm above the 0.79 homology-precision plateau of ESM2-8M.
- The tm_hi / tm_lo / blk650 label sets were chosen as ESM anchor pairs and favour arms
  built on ESM cosine; sd_invis is the unbiased check and agrees in direction.

## 2026-11-11 08:16 MST: hand-off state

Still running at hand-off: 76395 (ESM2-650M genome-centred graph, 64 CPUs; IVF 20-NN search
at 2 h 17 min, 135 CPU-hours so far, against 26 CPU-hours for ESM2-8M). Queued behind it:
76462 (quality with TM for `graph__esm2_650m__genomecentred__mknn20r{100,300,1000,3000}`,
afterok 76395) and 76463 (`run_finalize.sbatch`: recall for every arm, `quality.tsv`,
`recall_strata.tsv`; afterany 76462). Their rows will appear in `$RD/ws_tokens/quality.tsv`
without further action; the local mirror in `results/` predates them.
Local `code/` matches remote `$RD/ws_tokens/code/` byte for byte (sha256, 07:20 MST).

## 2026-11-11 09:05 MST: ESM2-650M graph arm (closes the hand-off item above)

Job 76395 finished in 2 h 56 min on 64 CPUs (IVF 20-NN search 10,217 s, peak RSS 50 GB):
4,181,264 nodes, 21,240,201 mutual edges, 156,551 isolated, mean degree 10.16. Quality 76462.

| graph 650m gnm | tokens | max | sd_vis | sd_invis | blk650_vis | tm_hi | tm_lo | random | seqprec pair | TMconf | homology prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r100 | 195,594 | 8,497 | 0.390 | 0.146 | 0.910 | 0.495 | 0.038 | 1.5e-4 | 0.408 | 0.368 | 0.626 |
| r300 | 202,928 | 3,916 | 0.292 | 0.074 | 0.893 | 0.444 | 0.032 | 6e-5 | 0.526 | 0.421 | 0.726 |
| r1000 | 216,409 | 865 | 0.237 | 0.041 | 0.858 | 0.356 | 0.028 | 2e-5 | 0.639 | 0.483 | **0.813** |
| r3000 | 238,208 | 372 | 0.197 | 0.024 | 0.797 | 0.279 | 0.022 | 2e-5 | 0.710 | 0.508 | **0.857** |

- The larger model lifts the graph arm past the ESM2-8M plateau (0.79): r1000 and r3000 clear
  the 0.80 homology gate, the only embedding-only arms that do.
- Its gains sit on the ESM-selected label sets (tm_hi 0.28-0.36, blk650_vis 0.80-0.86); on
  the unbiased Spacedust sets it trails the CPM hybrid r = 0.05 (sd_vis 0.197-0.237 vs 0.799,
  sd_invis 0.024-0.041 vs 0.142) at lower homology precision (0.81-0.86 vs 0.966).
  The recommendation stands.
- Under the predeclared rule restricted to embedding-only arms, the pick is
  `graph__esm2_650m__genomecentred__mknn20r1000` (tm_hi 0.356 at homology precision 0.813).

## 2026-11-11 09:15 MST: final state

`run_finalize` 76463 completed: `$RD/ws_tokens/quality.tsv` has 56 rows (all arms of this
workstream), `recall_strata.tsv` covers every arm; both mirrored to `results/` with all
manifests and per-arm `stats.json` / `recall.tsv`. No jobs of this workstream remain queued.
