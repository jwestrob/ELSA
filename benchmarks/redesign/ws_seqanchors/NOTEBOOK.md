# ws_seqanchors notebook

Append-only. Every entry carries a timestamp (Pacific time, the biotite clock).
Remote directory: `$RD/ws_seqanchors/` with
`RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111`.

---

## 2026-11-11 03:15 PST. Question, design and predeclared criteria (written before any result)

**Question.** On the Spacedust collection (67,148 query proteins in 9 actinomycete
genomes; 4,190,383 reference proteins in 1,308 genomes), what do protein-embedding
anchors add over pairwise-sequence anchors, when both go through the same chainer
(published `chain.py`, sha256 `c7e40b8b...`, gap 2, minimum 2 anchors, paired-overlap
extraction), the same whole-pipeline permutation null (query gene order permuted with
`ordered_metadata`, seeds 17/18/19, before every order-dependent step) and the same
calibration (per block size 2, 3, 4, 5, 6+: the cut on mean anchor weight that holds
permuted/observed at 5%) and the same benchmark scoring (`source207`, 200 random
placements, seed 17, region bootstrap)?

**Arms.**

| arm | anchors | weight |
|---|---|---|
| `S-all/unit`, `S-all/bitrank` | every MMseqs2 hit, E <= 1e-3 | 1, or bit-score percentile |
| `S-top60/unit`, `S-top60/bitrank` | 60 best hits per query by bit score | 1, or bit-score percentile |
| `S-top60-cov50/unit`, `S-top60-cov50/bitrank` | same, after >= 50% coverage of both proteins | 1, or bit-score percentile |
| `E-ctx60/cos` | saved context60 anchors (replay of the published run) | raw cosine |
| `E-ctx60/rank` | same | cosine percentile |
| `E-seed60/cos`, `E-seed60/rank` | saved ANN60 seeds | raw cosine, or cosine percentile |
| `H-ctx60` | `S-top60` union context60 | percentile of each source |
| `H-seed60` | `S-top60` union ANN60 seeds | percentile of each source |

Embedding arms and hybrids for `esm2_8m`, `esm2_650m`, `esmc600m`.

**One weight scale (rank-based).** A sequence anchor gets G(bits), the empirical CDF
of bit score over all `S-top60` hits. An embedding anchor gets F_m(cosine), the
empirical CDF of cosine over model m's ANN60 seeds (both are "top-60 per query"
populations, so a percentile means the same thing on both sides). A pair present in
both sources keeps the larger weight (the chainer's own deduplication keeps the
strongest exact pair). Weights lie in (0, 1]. The calibration statistic is the mean
weight of a block's anchors. A calibrated probability of homology is the alternative;
it needs structure labels over the full score range and is left as a follow-up.

**Null.** Sequence hits and ANN60 seeds do not depend on gene order, so their null
chains the same anchors on permuted query positions. Context60 anchors were selected
with gene order, so under each permutation they are regenerated from the saved seeds
by the frozen extension kernel with permuted query neighbours, exactly as
`permuted_panels.py` does. Replay gates: `E-ctx60/cos` for `esm2_8m` must reproduce the
saved observed blocks and permuted totals 495,161 / 495,380 / 495,342; `esm2_650m` and
`esmc600m` must reproduce their totals in `$S/results/permuted_panels/<m>.json`.

**Predeclared readouts and criteria.** Primary comparison per model at the 5% level:
embedding arm `E-ctx60/rank` and hybrid `H-ctx60` against the budget-matched sequence
arm `S-top60/bitrank`. Secondary: against the union of accepted pairs over every
sequence arm.

- *Material contribution.* Embeddings contribute materially if accepted anchor pairs
  absent from the sequence arm number at least **10% of the sequence arm's accepted
  pairs**, or the accepted blocks of the embedding/hybrid arm cover at least **5% more
  query genes** (genes inside an accepted block's query span) than the sequence arm.
- *Quality of the addition.* At least **50%** of a sample of the embedding-unique
  accepted pairs are structure-confirmed (TM-score >= 0.5 normalized by the shorter
  chain, AFDB models, foldseek TM-align with exact TM-score).
- *Verdict.* Embeddings earn their place for a model only if both hold. If the
  sequence arm matches or exceeds the embedding arm on accepted pairs, covered query
  genes, `source207` F1 excess and cross-phylum accepted blocks, the result is stated
  as: sequence anchors suffice on this collection, and embeddings add nothing
  measurable. If the addition is material but under 50% confirmed, the result is
  stated as: embeddings add pairs, and most of them lack structural support.
- Controls reported beside the structure share: the confirmed share of
  sequence-unique and of shared accepted pairs (validates the readout).

**Known asymmetry, reported and kept.** ESM2 models carry no vector for proteins over
1,022 residues (936 query; 75,331 reference for 35M-650M). Sequence arms cover them.
The decisive table reports how many sequence-unique pairs involve such proteins.

## 2026-11-11 03:18 PST. Jobs launched

Private code copies: `$RD/ws_seqanchors/code/scale_code/` (copy of `$SCALE/code`) and
`$RD/ws_seqanchors/code/production/elsa/` (copy of `$SCALE/production/elsa`; `chain.py`
sha256 `c7e40b8b2a5d2816e3412bee9e346827f3842fa4d9cf4acfcde708733f70d9c9`, identical).
Local mirror: `code/` here.

- **Job 76256** `code/run_search.sbatch` (`standard_expand_2`, 64 CPU, node-64-768g-11,
  Xeon E5-2683 v4). MMseqs2 release 18 (`8cc5ce36...`):
  `prefilter querydb keggclusterdb -s 7.5 --max-seqs 4000`, then
  `align -e 1e-3 --alignment-mode 3` (composition-bias correction on, the default, as
  tonight), `prefixid --tsv` (results read by key), `result2stats --stat linecount` on the
  prefilter (to count queries whose candidate list hit the 4,000 cap). Then
  `code/search_post.py`: keys -> row indices through `$RD/universe/genes.parquet`,
  per-query rank by bits (ties: E-value, then target row), coverage
  `(end-start+1)/len` on both sides, writes `work/hits.parquet`,
  `work/bits_top60_sorted.npy`, `results/search_stats.json`. Smoke-tested on 200,000
  lines of tonight's `aln_seq.tsv` (E <= 1e-3) before submission.
  `S-all` is therefore "every hit at E <= 1e-3 among each query's 4,000 best prefilter
  candidates"; the number of queries at the cap is recorded.
- **Job 76259** (array 0-2: `esm2_8m`, `esm2_650m`, `esmc600m`) `code/run_emb.sbatch` ->
  `code/sa.py emb`. Context60 extension replayed on the observed order (must equal the
  saved `anchors.parquet` pair for pair), then regenerated for permutation seeds 17/18/19
  (`work/ctx_perm/<m>_seed<s>.parquet`); arms `E-ctx60/{cos,rank}` and
  `E-seed60/{cos,rank}`, each chained on the observed order and the three permutations,
  then calibrated and scored. `E-ctx60/cos` must replay the published blocks (hard
  assert) and tonight's permuted totals (checked and logged).

Bootstrap note: `permuted_panels.py` drew all its region bootstraps from one
`default_rng(17)` stream in call order; `sa.py` seeds a fresh `default_rng(17)` per
scored selection, so its intervals are reproducible per selection and can differ from
tonight's in the third decimal.

## 2026-11-11 03:30 PST. Dependent jobs queued

- **76280** `code/run_stop60.sbatch` (24 CPU, after 76256 and 76259): `S-top60/{unit,bitrank}`,
  `S-top60cov50/{unit,bitrank}`.
- **76281** `code/run_hyb.sbatch` (array 0-2, at most 2 running, after 76256 and 76259):
  `H-ctx60@m`, `H-seed60@m`.
- **76282** `code/run_sall.sbatch` (`standard_expand_2`, 64 CPU, after 76256): `S-all/{unit,bitrank}`.
- **76283** `code/run_compare.sbatch` (`standard_expand_2`, 64 CPU, after 76280/76281/76282):
  `code/compare.py prep` (arm tables, decisive counts, structure sample), foldseek
  `tmalign` + `convertalis --exact-tmscore 1` on aliased query/target DBs whose headers
  are `K<key>`, then `compare.py summarize`.

Structure sample, fixed now: for `esm2_650m` and `esmc600m`, primary comparison
`E-ctx60/rank` vs `S-top60/bitrank`: up to 15,000 embedding-only accepted pairs, 4,000
shared and 3,000 sequence-only, each the first N by `hash(qkey*5e6+tkey)` among pairs
with AFDB models on both sides (<= 44,000 pairs). TM-scores already computed tonight
(`$S/work_key/tm_key.parquet`, `$S/work/pair_table.parquet`) are used where present.

Early replay evidence (03:20): `esm2_8m` context60 under seed 17 gives 495,161 blocks, the
saved total; `esm2_650m` (3,971,948 anchors) and `esmc600m` (4,027,458 anchors) context60
extension replays are identical pair for pair to the saved `anchors.parquet`.

## 2026-11-11 03:32 PST. Job 76259 complete: replay gates pass, embedding arms done

Replay gates (all exact):

| model | observed blocks / saved | anchor pairs / saved | permuted totals (17/18/19) | tonight's panel | September saved |
|---|---|---|---|---|---|
| esm2_8m | 534,990 / 534,990 | 1,482,427 / 1,482,427 | 495,161 / 495,380 / 495,342 | match | match |
| esm2_650m | 478,610 / 478,610 | 1,513,349 / 1,513,349 | 318,561 / 317,610 / 319,935 | match | n/a |
| esmc600m | 463,807 / 463,807 | 1,377,536 / 1,377,536 | 368,199 / 368,589 / 368,952 | match | n/a |

Context60 extension replays on the observed order are identical pair for pair to the
saved `anchors.parquet` (3,970,730 / 3,971,948 / 4,027,458 anchors). Calibrated
`E-ctx60/cos` reproduces tonight's numbers: accepted blocks at 5% 111,389 (8M) and
265,639 (650M); F1 0.652 / 0.754 / 0.762; 8M cross-phylum accepted blocks 4,474 (brief F).
Extension wall time per run 36-72 s on 24 CPUs.

Embedding arms at 5% (from the job logs; full tables follow in RESULTS.md):

| arm | observed | perm/obs | accepted blocks | accepted pairs | F1 cal | excess | cross-phylum |
|---|---|---|---|---|---|---|---|
| E-ctx60/cos@esm2_8m | 534,990 | 0.926 | 111,389 | 536,198 | 0.652 | -0.006 | 4,474 |
| E-ctx60/rank@esm2_8m | 535,028 | 0.926 | 112,967 | 539,041 | 0.636 | -0.021 | 4,018 |
| E-seed60/cos@esm2_8m | 126,775 | 0.011 | 126,775 | 344,086 | 0.517 | -0.019 | 4,798 |
| E-seed60/rank@esm2_8m | 126,775 | 0.011 | 126,775 | 344,083 | 0.517 | -0.019 | 4,798 |
| E-ctx60/cos@esm2_650m | 478,610 | 0.666 | 265,639 | 1,052,580 | 0.754 | -0.005 | 17,075 |
| E-ctx60/rank@esm2_650m | 478,696 | 0.666 | 279,366 | 1,086,670 | 0.754 | -0.008 | 17,544 |
| E-seed60/cos@esm2_650m | 225,862 | 0.009 | 225,862 | 642,321 | 0.578 | -0.004 | 11,408 |
| E-seed60/rank@esm2_650m | 225,864 | 0.009 | 225,864 | 642,319 | 0.578 | -0.004 | 11,408 |
| E-ctx60/cos@esmc600m | 463,807 | 0.795 | 147,858 | 683,194 | 0.762 | +0.048 | 3,840 |
| E-ctx60/rank@esmc600m | 463,893 | 0.795 | 146,888 | 678,435 | 0.742 | +0.032 | 3,787 |
| E-seed60/cos@esmc600m | 168,577 | 0.019 | 168,577 | 482,221 | 0.595 | +0.020 | 3,744 |
| E-seed60/rank@esmc600m | 168,580 | 0.019 | 168,580 | 482,209 | 0.595 | +0.021 | 3,744 |

Observations. Seeds alone (no context selection) give permuted/observed 0.009-0.019 and
every block passes the 5% cut; their F1 is 0.10-0.17 lower than context60 because they
make fewer, shorter blocks over the annotated regions, and their F1 excess over random
placement equals or beats context60 for ESM2 (-0.004 vs -0.008 at 650M). The percentile
weight changes accepted blocks by -0.7% to +5.2% relative to raw cosine and F1 by 0.000
to -0.020; the published cosine scale stays the better-scoring one for context60 arms.

Correction (03:34): the clause "because they make fewer, shorter blocks over the annotated
regions" in the entry above is a hypothesis; only "fewer blocks" (126,775 vs 534,990 for
8M) is measured. Block length over the regions has not been checked.

## 2026-11-11 03:42 PST. Job 76256 complete: sequence search

MMseqs2 release 18, 64 CPUs (Xeon E5-2683 v4, node-64-768g-11): prefilter 700 s, align
846 s, export 23 s; search wall time 1,546 s (25.8 min), 98,944 CPU-seconds nominal.
Post-processing 29 s. `results/search_stats.json`.

| quantity | value |
|---|---|
| hits at E <= 1e-3 (`S-all`) | 63,080,078 |
| queries with >= 1 hit | 63,013 of 67,148 (93.8%) |
| hits per query, quantiles 0.1/0.25/0.5/0.75/0.9 | 3 / 42 / 346 / 1,311 / 3,363 |
| `S-top60` | 3,176,472 (48,403 queries have >= 60 hits) |
| `S-top60-cov50` | 2,922,419 (45,789,605 hits pass cov50; 43,610 queries have >= 60) |
| median identity, top60 / all | 0.447 / 0.323 |
| prefilter candidates | 268,212,859; 66,914 queries reach the 4,000 cap |
| queries with all 4,000 candidates passing E <= 1e-3 | 3,356 |

Truncation check. For the 3,356 queries whose 4,000 candidates all pass, the S-all list is
truncated by the cap (paralog-rich families). Their 4,000th-best hit scores a median 0.28
(IQR 0.21-0.35, max 0.73) of the rank-60 bit score, so the cap cuts far below the top 60:
`S-top60` and `S-top60-cov50` are complete to the precision this check allows. `S-all` is
"every E <= 1e-3 hit among the 4,000 best prefilter candidates".

For scale: the embedding arms' retrieval used precomputed vectors whose reference
inference took 58,370 s (ESM2-650M) and 11,545 s (ESM-C 600M) of committed embedding time
(manifests), plus an HNSW build of 996 s / 1,840 s and queries of 17 s / 31 s on 64 workers.

Jobs 76280 (S-top60 arms), 76281_0/1 (hybrids 8M, 650M) and 76282 (S-all) started 03:41.

## 2026-11-11 03:52 PST. Structure-sample plan amended (before any TM-score was computed)

The first hybrid results show `H-seed60` (S-top60 union ANN60 seeds) accepting 18% more
blocks than `S-top60` with a clean null (8M: 335,453 blocks, permuted/observed 0.011). Its
embedding-added pairs are therefore a second, cleaner candidate for "what embeddings add",
beside `E-ctx60/rank`. The sample is rebalanced, total still <= 50,000: per model
(`esm2_650m`, `esmc600m`) 10,000 `E-ctx60/rank`-only pairs, 2,500 shared, 2,500
`S-top60/bitrank`-only, and 8,000 `H-seed60` accepted pairs that are absent from both the
sequence arm's accepted pairs and the S-top60 anchors (`hseed_emb_added`). The 50%
criterion is applied to `E-ctx60/rank`-only pairs (as predeclared) and reported for
`hseed_emb_added` too.

Results so far at 5% (logs):

| arm | observed | perm/obs | accepted blocks | accepted pairs | F1 cal | excess | cross-phylum |
|---|---|---|---|---|---|---|---|
| S-top60/unit | 283,549 | 0.005 | 283,549 | 866,264 | 0.641 | +0.055 | 35,854 |
| S-top60/bitrank | 283,116 | 0.005 | 283,116 | 865,281 | 0.647 | +0.060 | 35,792 |
| S-top60cov50/unit | 268,838 | 0.005 | 268,838 | 814,161 | 0.621 | +0.048 | 33,272 |
| H-ctx60@esm2_8m | 829,245 | 0.974 | 245,957 | 1,011,738 | 0.735 | +0.026 | 22,811 |
| H-ctx60@esm2_650m | 906,529 | 1.119 | 298,772 | 1,249,355 | 0.781 | +0.030 | 28,537 |
| H-seed60@esm2_8m | 335,453 | 0.011 | 335,453 | 1,014,998 | 0.668 | +0.062 | 41,531 |

## 2026-11-11 03:58 PST. More arms; compare smoke test

- Job 76280 complete. `S-top60cov50/bitrank`: 268,740 blocks, permuted/observed 0.005, all
  accepted, 813,958 pairs, F1 0.623, excess +0.050, cross-phylum 33,271.
- `H-seed60@esm2_650m`: 367,290 blocks, permuted/observed 0.010, all accepted, 1,107,529
  pairs, F1 0.674, excess +0.062, cross-phylum 44,744.
- `S-all/unit` (job 76282, 64 CPU): 63,080,078 anchors, 3,696,665 observed blocks; seed 17
  gives 118,821 permuted blocks. Chain 291 s observed, 193 s per permutation, 46 GB RSS.
- **Job 76355** `code/run_compare_smoke.sbatch` (24 CPU): `compare.py prep` on the arms
  finished at 03:50 (S-all and the ESM-C hybrids absent). It ran clean in 35 s; outputs are
  overwritten by job 76283. First decisive counts (5% level, reference `S-top60/bitrank`,
  865,281 accepted pairs):

| arm | accepted pairs | shared | arm-only | arm-only / ref | arm-only in search (E<=1e-3) | arm-only with no hit |
|---|---|---|---|---|---|---|
| E-ctx60/rank@esm2_650m | 1,086,670 | 534,141 | 552,529 | 63.9% | 231,154 | 321,375 |
| H-ctx60@esm2_650m | 1,249,355 | 769,549 | 479,806 | 55.5% | 224,009 | 255,797 |
| E-seed60/rank@esm2_650m | 642,319 | 455,591 | 186,728 | 21.6% | 146,662 | 40,066 |
| H-seed60@esm2_650m | 1,107,529 | 863,304 | 244,225 | 28.2% | 197,416 | 46,809 |
| E-ctx60/rank@esmc600m | 678,435 | 413,819 | 264,616 | 30.6% | 119,428 | 145,188 |
| E-seed60/rank@esmc600m | 482,209 | 359,737 | 122,472 | 14.2% | 99,246 | 23,226 |

  "In search" means the pair is an E <= 1e-3 hit of the 4,000-candidate search but sits
  outside the sequence arm's accepted blocks (mostly below the per-query top-60 bit-score
  budget: only 40,515 of the 231,154 for 650M context60 are in S-top60). The material
  threshold (>= 10% more accepted pairs) is passed by every embedding arm on raw counts; the
  verdict now rests on the structure readout and on the S-all reference.

## 2026-11-11 04:24 PST. All 24 arms complete; decisive counts (job 76283 prep)

Jobs 76280, 76281 (0-2), 76282 COMPLETED. `S-all` chaining: 292-319 s observed and about
193 s per permutation on 64 CPUs, 46 GB. Full tables: `results/arms.tsv`,
`results/arms_by_stratum.tsv`, `results/decisive_counts.tsv`.

Sequence arms and hybrids at 5% (embedding-only arms in the 03:32 entry):

| arm | anchors | observed | perm/obs | ratio size 2 | accepted blocks | accepted pairs | expected chance blocks | covered query genes | F1 cal | random | excess [95% CI] | cross-phylum (>=10 anchors) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-all/unit | 63,080,078 | 3,696,665 | 0.033 | 0.045 | 3,696,665 | 9,510,980 | 122,455 | 58,039 | 0.724 | 0.654 | +0.070 [+0.049, +0.091] | 2,502,534 (16,976) |
| S-all/bitrank | 63,080,078 | 3,675,859 | 0.033 | 0.046 | 3,675,859 | 9,463,514 | 122,325 | 58,062 | 0.727 | 0.654 | +0.073 [+0.052, +0.094] | 2,488,554 (16,975) |
| S-top60/unit | 3,176,472 | 283,549 | 0.005 | 0.008 | 283,549 | 866,264 | 1,423 | 49,277 | 0.641 | 0.586 | +0.055 [+0.021, +0.088] | 35,854 (27) |
| S-top60/bitrank | 3,176,472 | 283,116 | 0.005 | 0.008 | 283,116 | 865,281 | 1,423 | 49,335 | 0.647 | 0.587 | +0.060 [+0.026, +0.093] | 35,792 (27) |
| S-top60cov50/unit | 2,922,419 | 268,838 | 0.005 | 0.008 | 268,838 | 814,161 | 1,214 | 47,538 | 0.621 | 0.573 | +0.048 [+0.013, +0.082] | 33,272 (12) |
| S-top60cov50/bitrank | 2,922,419 | 268,740 | 0.005 | 0.008 | 268,740 | 813,958 | 1,215 | 47,558 | 0.623 | 0.574 | +0.050 [+0.015, +0.084] | 33,271 (12) |
| H-ctx60@esm2_8m | 6,567,031 | 829,245 | 0.974 | 1.167 | 245,957 | 1,011,738 | 9,826 | 59,754 | 0.735 | 0.709 | +0.026 [-0.001, +0.052] | 22,811 (40) |
| H-seed60@esm2_8m | 6,363,676 | 335,453 | 0.011 | 0.018 | 335,453 | 1,014,998 | 3,656 | 52,038 | 0.668 | 0.606 | +0.062 [+0.032, +0.091] | 41,531 (29) |
| H-ctx60@esm2_650m | 6,494,729 | 906,529 | 1.119 | 1.285 | 298,772 | 1,249,355 | 11,942 | 62,794 | 0.781 | 0.751 | +0.030 [+0.007, +0.052] | 28,537 (90) |
| H-seed60@esm2_650m | 5,815,345 | 367,290 | 0.010 | 0.017 | 367,290 | 1,107,529 | 3,738 | 52,966 | 0.674 | 0.612 | +0.062 [+0.032, +0.091] | 44,744 (44) |
| H-ctx60@esmc600m | 6,625,334 | 802,557 | 1.070 | 1.267 | 244,231 | 1,042,443 | 9,866 | 61,476 | 0.788 | 0.731 | +0.057 [+0.036, +0.077] | 20,611 (46) |
| H-seed60@esmc600m | 6,254,696 | 340,335 | 0.016 | 0.026 | 340,335 | 1,033,939 | 5,363 | 53,979 | 0.678 | 0.618 | +0.060 [+0.031, +0.089] | 39,164 (27) |

Embedding-only context60 arms for comparison: E-ctx60/rank@esm2_650m 279,366 accepted
blocks, 1,086,670 pairs, 64,708 covered genes, excess -0.008 [-0.031, +0.013], cross-phylum
17,544 (51); E-ctx60/rank@esmc600m 146,888 / 678,435 / 60,201 / +0.032 [+0.004, +0.057] /
3,787 (13).

Findings, stated plainly:
1. Every sequence arm has a clean null without context selection (permuted/observed 0.005
   for top-60, 0.033 for S-all; the size-2 class of S-all sits at 0.045, inside the 5% rule).
   Unit and bit-score weights give the same result (blocks within 0.6%, F1 excess within
   0.005): with a clean null the weight carries no calibration information.
2. `S-all` has the largest F1 excess of all 24 arms (+0.070 / +0.073) and finds 2.50 M
   accepted cross-phylum blocks (16,976 with >= 10 anchors), against 3,787-17,544 for the
   context60 embedding arms and 39,164-44,744 for the seed hybrids.
3. Adding context60 anchors to `S-top60` (H-ctx60) raises chance to the embedding level
   (size-2 ratio 1.17-1.29), so the 5% cut discards most small blocks, and the hybrid
   accepts *fewer* cross-phylum blocks (20,611-28,537) than `S-top60` alone (35,792).
4. Adding ANN60 seeds to `S-top60` (H-seed60) keeps a clean null (0.010-0.016) and adds
   18-30% accepted blocks and 17-28% accepted pairs; F1 excess is unchanged
   (+0.060-0.062 vs +0.060).

Decisive counts at 5% (reference `S-top60/bitrank`, 865,281 accepted pairs, 283,116
blocks, 49,335 covered query genes; secondary reference: union of all six sequence arms,
9,823,998 pairs, 58,065 covered genes):

| arm | ref | arm-only pairs | / ref | arm-only with no search hit | ref-only pairs | arm-only blocks | ref-only blocks | covered genes added by arm | added by ref |
|---|---|---|---|---|---|---|---|---|---|
| E-ctx60/rank@esm2_650m | S-top60 | 552,529 | 63.9% | 321,375 | 331,140 | 109,923 | 109,150 | 15,844 (+32.1%) | 471 |
| E-ctx60/rank@esm2_650m | union | 339,079 | 3.5% | 321,375 | 9,076,407 | - | - | 8,029 (+13.8%) | 1,386 |
| H-seed60@esm2_650m | S-top60 | 244,225 | 28.2% | 46,809 | 1,977 | 86,562 | 36 | 3,636 (+7.4%) | 5 |
| H-seed60@esm2_650m | union | 66,100 | 0.7% | 46,809 | 8,782,569 | - | - | 1,754 (+3.0%) | 6,853 |
| E-ctx60/rank@esmc600m | S-top60 | 264,616 | 30.6% | 145,188 | 451,462 | 31,628 | 163,886 | 12,412 (+25.2%) | 1,546 |
| E-ctx60/rank@esmc600m | union | 151,204 | 1.5% | 145,188 | 9,296,767 | - | - | 6,134 (+10.6%) | 3,998 |
| H-seed60@esmc600m | S-top60 | 170,771 | 19.7% | 27,763 | 2,113 | 59,138 | 48 | 4,645 (+9.4%) | 1 |
| H-seed60@esmc600m | union | 38,531 | 0.4% | 27,763 | 8,828,590 | - | - | 2,348 (+4.0%) | 6,434 |

Against the union of sequence arms, almost every embedding-only accepted pair is one the
search does not return at all (650M context60: 321,375 of 339,079). Against `S-top60`, most
seed-hybrid additions are pairs the search does return but outside each query's top 60 by
bit score (197,416 of 244,225 for 650M).

Expected chance content of the context60 accepted sets is 3.9% of blocks (650M:
10,950 of 279,366), so part of the coverage they add is chance; the structure readout
measures the pair-level share.

**Job 76377** `code/run_paired.sbatch` (`compare.py paired`): paired region bootstrap
(10,000 draws, `default_rng(17)`) of the per-region F1-excess difference between each
embedding/hybrid arm and `S-top60/bitrank` or `S-all/bitrank`; control placements are
identical across arms (seeded by region), so the pairing is exact.

## 2026-11-11 04:30 PST. Job 76377 complete: paired F1-excess differences

`results/paired_f1_excess.tsv`, `results/per_region_excess.tsv`. Difference = arm excess
minus reference excess, mean over 207 regions, 95% paired region-bootstrap interval.

| arm | vs S-top60/bitrank (+0.060) | vs S-all/bitrank (+0.073) |
|---|---|---|
| S-all/bitrank | +0.013 [-0.010, +0.037] | - |
| E-ctx60/cos@esm2_8m | -0.066 [-0.100, -0.033] | -0.079 [-0.111, -0.048] |
| E-ctx60/cos@esm2_650m | -0.065 [-0.095, -0.035] | -0.078 [-0.102, -0.055] |
| E-ctx60/rank@esm2_650m | -0.068 [-0.098, -0.037] | -0.081 [-0.105, -0.057] |
| E-seed60/rank@esm2_650m | -0.064 [-0.094, -0.036] | -0.077 [-0.108, -0.047] |
| H-ctx60@esm2_650m | -0.030 [-0.051, -0.008] | -0.042 [-0.065, -0.020] |
| H-seed60@esm2_650m | +0.002 [-0.009, +0.016] | -0.011 [-0.030, +0.007] |
| E-ctx60/cos@esmc600m | -0.012 [-0.040, +0.016] | -0.025 [-0.050, -0.002] |
| E-ctx60/rank@esmc600m | -0.028 [-0.053, -0.003] | -0.041 [-0.066, -0.017] |
| E-seed60/rank@esmc600m | -0.039 [-0.064, -0.015] | -0.052 [-0.082, -0.025] |
| H-ctx60@esmc600m | -0.003 [-0.026, +0.021] | -0.016 [-0.033, +0.002] |
| H-seed60@esmc600m | +0.001 [-0.012, +0.015] | -0.012 [-0.033, +0.007] |
| H-seed60@esm2_8m | +0.002 [-0.010, +0.017] | -0.011 [-0.030, +0.007] |
| H-ctx60@esm2_8m | -0.034 [-0.054, -0.012] | -0.047 [-0.069, -0.024] |

Result: on `source207`, no embedding or hybrid arm beats either sequence arm. Every
embedding-only arm is worse than `S-top60` except ESM-C 600M context60 at raw cosine
(interval spans 0); every embedding-only arm is worse than `S-all`. Context60 hybrids are
worse than or equal to their sequence base; seed hybrids equal it to within +-0.017.

## 2026-11-11 04:52 PST. Job 76283 complete: structure readout and verdict

foldseek `tmalign` (10-941cd33) + `convertalis --exact-tmscore 1` on 31,495 unique pairs
(46,000 sample rows; the two models' sets overlap), 64 CPUs, 25.4 min + 1 min.
`$S/work_key/tm_key.parquet` had not landed when the sample was drawn, so every pair was
aligned here. Cross-check after it landed: 11,303 pairs are in both tables, with
identical TM-scores (max |difference| 0.0) and identical calls. Control from tonight's key
subset: uniform random query x reference pairs are 1.82% confirmed (n = 23,788).
`results/structure_confirmation.tsv`; per-pair `work/tm/sample_scored.parquet` (remote).

Structure-confirmed = max(qTM, tTM) >= 0.5 (TM normalized by the shorter chain); Wilson 95%.
"in search" = the pair is an E <= 1e-3 hit of the 4,000-candidate search; "no hit" = it is not.

| model | set | n | confirmed | in search: n, confirmed | no hit: n, confirmed |
|---|---|---|---|---|---|
| esm2_650m | E-ctx60/rank only (vs S-top60) | 10,000 | 52.4% [51.4, 53.4] | 4,336, 97.6% | 5,664, 17.8% [16.8, 18.8] |
| esm2_650m | H-seed60 embedding-added | 8,000 | 89.7% [89.0, 90.4] | 5,967, 97.1% | 2,033, 68.0% [66.0, 70.0] |
| esm2_650m | shared (E-ctx60 and S-top60) | 2,500 | 96.8% | all | - |
| esm2_650m | S-top60 only | 2,500 | 96.9% | all | - |
| esmc600m | E-ctx60/rank only (vs S-top60) | 10,000 | 51.7% [50.7, 52.6] | 4,612, 98.1% | 5,388, 11.9% [11.1, 12.8] |
| esmc600m | H-seed60 embedding-added | 8,000 | 88.0% [87.2, 88.7] | 6,319, 98.2% | 1,681, 49.7% [47.3, 52.1] |
| esmc600m | shared | 2,500 | 97.8% | all | - |
| esmc600m | S-top60 only | 2,500 | 96.2% | all | - |

Stratified estimates for the additions over the union of all sequence arms (counts from the
04:24 entry, rates from the strata above):
- E-ctx60/rank@esm2_650m: 339,079 pairs = 17,704 in search + 321,375 no hit, est. 22.0%
  confirmed (about 74,500 pairs). esmc600m: 151,204 = 6,016 + 145,188, est. 15.3%
  (about 23,200).
- H-seed60@esm2_650m: 66,100 = 19,291 + 46,809, est. 76.5% confirmed (about 50,600; about
  31,800 of them with no sequence hit). esmc600m: 38,531 = 10,768 + 27,763, est. 63.3%
  (about 24,400; about 13,800 with no hit). Approximation: the H-seed60 sample excludes pairs
  in the S-top60 anchor set, which are 14,049 / 7,690 of these additions.

**Verdict against the predeclared criteria.**

Primary (`E-ctx60/rank` vs `S-top60/bitrank`): material (650M +63.9% pairs, +32.1% covered
genes; 600M +30.6%, +25.2%) and confirmed at 52.4% and 51.7%, both lower bounds above 50%.
The criterion is met, narrowly. The confirmed half is pairs the sequence search already
returns at E <= 1e-3 and the top-60 budget left out (97.6-98.1% confirmed); the half with
no sequence hit is 17.8% / 11.9% confirmed.

Secondary (vs the union of sequence arms, which includes `S-all`): additions are 3.5% /
1.5% of accepted pairs (under the 10% bar) and +13.8% / +10.6% covered genes (over the 5%
bar), estimated 22% / 15% confirmed (under the 50% bar). Stated as predeclared: embeddings
add pairs, and most of them lack structural support.

The seed hybrid is the form in which embeddings add something real: against `S-top60` it
adds 28.2% / 19.7% accepted pairs at 89.7% / 88.0% confirmation with a clean null; against
the union it adds 0.7% / 0.4% of pairs (under the bar), of which an estimated 31,800 /
13,800 pairs are structure-confirmed homologs with no sequence hit. On `source207` no
embedding or hybrid arm beats either sequence arm (paired differences, 04:30 entry).

## 2026-11-11 05:05 PST. Report written; workstream closed for tonight

`REPORT.md` holds the ablation table (24 arms), strata, paired F1 differences, the
decisive table, the structure readout, wall time, what works and what does not, artifact
paths and open questions. Small results are mirrored to `results/` and code to `code/`
(sha256 identical to `$RD/ws_seqanchors/code/`). Large files stay on biotite under
`$RD/ws_seqanchors/work/`, including the intermediate MMseqs databases in `work/search/`.
All jobs (76256, 76259, 76280, 76281, 76282, 76283, 76355, 76377) ended COMPLETED; none of
this workstream's jobs remain queued.

---

## 2026-11-11 08:00 PST. Follow-up: whitened-embedding arms. Question, design, predeclared criteria (before any result)

**Why.** The metric workstream (`ws_metric/NOTEBOOK.md`, closing report 16:35Z) found that raw
cosine is a poorly calibrated similarity. On 22,621 held-out queries, exact top-60 retrieval
with soft-ZCA whitening (alpha 0.1) raises the sequence-visible fraction of hits from 0.612
to 0.726 (esm2_650m) and from 0.388 to 0.651 (esmc600m). For other-phylum targets the rise
is 0.409 to 0.588 and 0.143 to 0.514. The learned 256-d head on the whitened input
(`zlin256_t10_vis_inb`) beats whitening on esmc600m only. The question "what do embeddings
add over sequence" is re-asked with these vectors.

**Inputs.** `$RD/ws_metric/vectors/{esm2_650m,esmc600m}__zca10.npy`
(gid order, unit rows; 76,267 NaN rows for esm2_650m, 0 for esmc600m) and
`esmc600m__zlin256_t10_vis_inb.npy` (256-d, unit rows, 0 NaN).

**Arms (all 67,148 queries, gap 2, min 2, paired overlap, same null and calibration as before).**

| arm | anchors | weight |
|---|---|---|
| `Z-seed60@m` | exact top-60 inner product over the whole reference, zca10 vectors, no similarity cut | percentile of its own similarity distribution |
| `HZ-seed60@m` | `S-top60` union `Z-seed60` | G(bits) and the Z percentile; a shared pair keeps the larger |
| `L-seed60@esmc600m`, `HL-seed60@esmc600m` | the same with the 256-d learned head | the same |
| `R-seed60@m` (control) | exact top-60 with the raw vectors, no cut | percentile |

`R-seed60` is added because `E-seed60` differs from `Z-seed60` in two ways: the HNSW
approximate search with a cosine >= 0.85 cut, and raw versus whitened space. `R-seed60` vs
`Z-seed60` isolates the metric. Queries and reference genomes are disjoint, so every hit
is cross-genome; this is asserted. Rows without a vector (esm2_650m) are left out of both
sides. Search is faiss 1.13.2 `knn` (`METRIC_INNER_PRODUCT`, exact) on one 64-CPU job, with
the wall time recorded per search.

**Replay gates.** On the metric workstream's 22,621 held-out queries, my `Z-seed60`,
`R-seed60` and `L-seed60` top-60 sets must equal its `hits/{zca10,raw,zlin256_t10_vis_inb}`
(search `all`). The tolerance is per-query Jaccard >= 0.999 overall (rank-60 float ties
allowed) and max |score difference| <= 1e-5.

**Null.** Exact top-60 retrieval does not depend on gene order. Each arm is chained on the
observed order and on the three permuted orders (seeds 17/18/19) with identical anchors.

**Predeclared criterion (decisive arm `HZ-seed60`, each model).** Whitened embeddings change
the verdict if `HZ-seed60` adds over the union of the six sequence arms (9,823,998 accepted
pairs; 58,065 covered query genes) **at least 2% accepted pairs (>= 196,480) or at least 5%
covered query genes (>= 2,903)**, with **at least 50% of the no-hit additions
structure-confirmed** (TM >= 0.5 by the shorter chain, Wilson interval reported). They leave
the verdict unchanged otherwise. `HL-seed60@esmc600m` is judged by the same rule as a
secondary arm.

**Expectation for cross-phylum blocks (written now).** Whitening raised the other-phylum
visible fraction most, so:
- `Z-seed60` should accept more cross-phylum blocks than raw seeds (`E-seed60`: 11,408 for
  650M, 3,744 for 600M): about 1.3-2x for esm2_650m and 2-5x for esmc600m.
- `HZ-seed60` should exceed `H-seed60` (44,744 / 39,164) by at most 30%.
- All stay below 2% of `S-all`'s 2.49 M cross-phylum blocks. The top-60 budget, not the
  metric, is what limits distant reach (BRIEF F).

**Structure sample (<= 20,000 pairs).** For `HZ-seed60@esm2_650m`, `HZ-seed60@esmc600m` and
`HL-seed60@esmc600m`: 4,500 no-hit additions over the union, 1,000 in-search additions over
the union, and 1,000 in-search additions over `S-top60`. Each set is the first N by
`hash(qkey*5e6+tkey)` among pairs with both structures. Scores come from
`$S/work/pair_table.parquet`, `$S/work_key/tm_key.parquet` and my
`work/tm/sample_scored.parquet` where present; foldseek TM-aligns the rest.

Correction: the entry above was written at 16:13 PST (biotite clock), not 08:00.

## 2026-11-11 16:25 PST. Whitened arms: jobs launched

Preflight repeated in this turn: both cluster files read in full (unchanged since the first read).

- **77597** `code/run_zsearch.sbatch` (`standard_expand_2`, 64 CPU, node-64-768g-12; node-64-768g-29
  excluded for the stalled reads ws_metric saw): `code/zsearch.py`. Five exact top-60 searches,
  faiss 1.13.2 `knn` (`METRIC_INNER_PRODUCT`), queries in blocks of 8,192, with whole-matrix
  sequential reads (21.5 GB reference read in 30 s). Spaces: `zca10__esm2_650m`,
  `zca10__esmc600m`, `zlin256__esmc600m`, `raw__esm2_650m`, `raw__esmc600m`. Each search writes
  `work/zseeds/<space>__<model>.parquet` and `results/zsearch/<name>.json` (timings, rows without
  vectors, norm checks, replay against ws_metric hits, ANN60 recall for raw). It was smoke-tested
  on the login node first (30,000 reference rows, 500 queries, 2 threads).
- **77598** `code/run_zarms.sbatch` (array 0-2, 24 CPU, after 77597): `sa.py arms` with the new arm
  kinds (`Z`, `L`, `R`, `HZ`, `HL`; `sa.py` gains `Ctx.exact`, the percentile weight over each
  space's own hits).
- **77617** `code/run_compare_z.sbatch` (`standard_expand_2`, 64 CPU, after 77598; it starts
  after 77597 ends, so only one 64-CPU job runs at a time): `code/compare_z.py` prep, foldseek
  on the pairs without a score, summarize (structure, verdict), paired F1 bootstrap.

## 2026-11-11 17:14 PST. Job 77597 complete: exact searches; replay gates pass

`results/zsearch/<name>.json`. 64 CPUs (Xeon E5-2683 v4), faiss 1.13.2 exact `knn`.

| space | dim | reference rows | queries | read s | search s | hits | replay: identical queries / 22,621 | mean Jaccard | max score diff |
|---|---|---|---|---|---|---|---|---|---|
| zca10 esm2_650m | 1,280 | 4,115,052 | 66,212 | 30 | 742 | 3,972,720 | 22,548 | 0.99983 | 3.0e-7 |
| zca10 esmc600m | 1,152 | 4,190,383 | 67,148 | 27 | 707 | 4,028,880 | 22,566 | 0.99990 | 3.0e-7 |
| zlin256 esmc600m | 256 | 4,190,383 | 67,148 | 6 | 186 | 4,028,880 | 22,538 | 0.99987 | 1.1e-6 |
| raw esm2_650m | 1,280 | 4,115,052 | 66,212 | 30 | 760 | 3,972,720 | 22,514 | 0.99984 | 5.4e-7 |
| raw esmc600m | 1,152 | 4,190,383 | 67,148 | 27 | 690 | 4,028,880 | 22,559 | 0.99990 | 6.0e-7 |

Replay against the metric workstream's hits passes for every space (the queries that differ
differ at tied rank-60 scores). The raw exact top-60 contains 99.82% (650M: 3,964,759 of
3,971,948) and 99.85% (600M: 4,021,218 of 4,027,458) of the HNSW ANN60 seeds. Only 772 / 1,413
exact raw hits fall below cosine 0.85, so `R-seed60` and `E-seed60` share almost every
anchor, and `Z` vs `R` isolates the metric. Similarity medians: raw 0.988 / 0.973, zca10
0.769 / 0.601, zlin256 0.835.

Job 77598 (arms) started 17:12; 77617 is pending on it.

## 2026-11-11 17:35 PST. Whitened arms complete (77598, 77617): results and verdict

Jobs 77598 (array 0-2) and 77617 COMPLETED. Files: `results/zarms.tsv`, `zarms_by_stratum.tsv`,
`zdecisive_counts.tsv`, `zstructure_confirmation.tsv`, `zverdict.tsv`, `zpaired_f1_excess.tsv`,
`zper_region_excess.tsv`, `tmz_prep_stats.json`, `arms/{Z,HZ,L,HL,R}-seed60__*.json`.

Arms at 5% (every exact-search arm has a clean null and accepts every block):

| arm | observed = accepted | perm/obs | accepted pairs | covered genes | F1 | random | excess [95% CI] | cross-phylum (>= 10) |
|---|---|---|---|---|---|---|---|---|
| R-seed60@esm2_650m | 226,051 | 0.009 | 642,978 | 49,007 | 0.578 | 0.582 | -0.004 [-0.038, +0.029] | 11,422 (13) |
| Z-seed60@esm2_650m | 284,974 | 0.006 | 823,910 | 49,860 | 0.602 | 0.590 | +0.012 [-0.021, +0.044] | 30,360 (38) |
| HZ-seed60@esm2_650m | 374,066 | 0.008 | 1,124,839 | 52,131 | 0.668 | 0.607 | +0.061 [+0.032, +0.090] | 53,305 (60) |
| R-seed60@esmc600m | 168,675 | 0.019 | 482,641 | 48,858 | 0.599 | 0.575 | +0.023 [-0.009, +0.055] | 3,757 (0) |
| Z-seed60@esmc600m | 276,950 | 0.013 | 810,326 | 51,902 | 0.639 | 0.604 | +0.035 [+0.004, +0.067] | 23,074 (30) |
| HZ-seed60@esmc600m | 366,165 | 0.013 | 1,102,548 | 53,393 | 0.673 | 0.615 | +0.058 [+0.028, +0.087] | 47,072 (50) |
| L-seed60@esmc600m | 313,821 | 0.006 | 942,932 | 51,174 | 0.645 | 0.601 | +0.044 [+0.011, +0.076] | 33,945 (44) |
| HL-seed60@esmc600m | 376,039 | 0.007 | 1,140,516 | 52,239 | 0.667 | 0.608 | +0.059 [+0.028, +0.090] | 53,887 (59) |

`R-seed60` reproduces the raw ANN60 arm (`E-seed60/rank`: 225,864 / 168,580 blocks; paired F1
difference -0.0003 / +0.003), so exact search and the 0.85 cut change nothing.

Paired F1-excess differences (95% region bootstrap):
- Whitening alone helps a seeds-only arm. Z vs exact raw: +0.016 [+0.003, +0.032] (650M) and
  +0.012 [-0.009, +0.035] (600M); L vs exact raw +0.021 [-0.002, +0.044]; L vs Z +0.009
  [-0.013, +0.029].
- Whitened seeds alone stay below sequence: Z vs `S-top60` -0.048 [-0.076, -0.020] (650M) and
  -0.024 [-0.043, -0.005] (600M); L -0.016 [-0.035, +0.003]; every Z/L arm is below `S-all`.
- Hybrids are unchanged by whitening. HZ vs `S-top60`: +0.001 [-0.011, +0.016] (650M), -0.002
  [-0.015, +0.014] (600M); HL -0.000 [-0.011, +0.013]. HZ vs the raw hybrid `H-seed60`: -0.001
  / -0.002; HL vs HZ +0.001. HZ/HL vs `S-all`: -0.011 to -0.014, all intervals spanning 0.

Decisive counts and structure (union of sequence arms = 9,823,998 accepted pairs, 58,065 genes):

| arm | additions over union (% of union) | in search / no hit | covered genes added | no-hit confirmed [95% CI] | in-search confirmed | est. confirmed additions | est. confirmed no-hit pairs |
|---|---|---|---|---|---|---|---|
| HZ-seed60@esm2_650m | 88,820 (0.90%) | 25,192 / 63,628 | +1,530 (+2.6%) | 70.4% [69.0, 71.7] | 94.0% | 77.1% (68,460) | 44,780 |
| HZ-seed60@esmc600m | 73,470 (0.75%) | 20,622 / 52,848 | +2,206 (+3.8%) | 62.1% [60.7, 63.5] | 93.9% | 71.0% (52,200) | 32,836 |
| HL-seed60@esmc600m | 98,032 (1.00%) | 28,102 / 69,930 | +1,593 (+2.7%) | 68.1% [66.7, 69.4] | 93.7% | 75.4% (73,946) | 47,615 |

Against `S-top60`, HZ adds 30.3% / 27.8% accepted pairs (HL 32.2%), +5.7% / +8.2% covered genes
(HL +5.9%), and in-search additions over `S-top60` are 97.3-97.8% confirmed. Raw-hybrid
comparison (03:52 and 04:52 entries): `H-seed60` added 66,100 / 38,531 pairs over the union,
no-hit confirmed 68.0% / 49.7%, est. 31,800 / 13,800 confirmed no-hit pairs. Whitening raises
the confirmed no-hit additions to 44,780 (x1.4) for 650M and 32,836 (x2.4) for 600M, and the
learned head to 47,615 (x3.5).

Structure sample: 19,500 rows, 11,482 unique pairs. 7,599 scores came from
`$S/work/pair_table.parquet` and 205 from this workstream's round-1 sample; 3,678 were
TM-aligned here (foldseek, 3.3 min on 64 CPUs). `$S/work_key/tm_key.parquet` contributed none
beyond those.

**Verdict (predeclared).** `HZ-seed60` adds 0.90% / 0.75% accepted pairs (bar 2%) and 2.6% /
3.8% covered query genes (bar 5%) over the union of sequence arms. The quality condition holds
(no-hit additions 70.4% / 62.1% confirmed, above 50%), and the materiality condition fails for
both models. **Whitened embeddings leave the verdict unchanged.** The secondary
`HL-seed60@esmc600m` gives the same answer: 1.00% pairs, 2.7% genes, 68.1% confirmed.

**Cross-phylum expectation vs result.**
- `Z-seed60` over raw seeds: x2.66 for 650M (30,360 vs 11,408) and x6.16 for 600M (23,074 vs
  3,744). Both exceed the predicted ranges (1.3-2x, 2-5x); whitening reaches distant genomes
  more than I expected.
- `HZ-seed60` over `H-seed60`: +19.1% (53,305 vs 44,744) and +20.2% (47,072 vs 39,164), inside
  the predicted <= 30%.
- Against `S-all/bitrank`'s 2,488,554 cross-phylum blocks: HZ is 2.14% (650M) and 1.89%
  (600M), HL 2.17%. The predicted "below 2%" holds for 600M only.

Correction: the entry above was written at 17:30 PST (biotite clock), not 17:35.

## 2026-11-11 17:40 PST. Whitened follow-up closed

`REPORT.md` gains the section "Whitened-embedding arms": findings, method, ablation rows,
strata, paired F1, decisive table, structure table, predeclared verdict, wall time and open
questions. Local `code/` holds 17 files with sha256 identical to the remote copies (the earlier
diff output differed only in sort order). Jobs 77597, 77598, 77617 COMPLETED; none of this
workstream's jobs remain queued.
