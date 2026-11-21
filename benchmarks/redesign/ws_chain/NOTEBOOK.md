# ws_chain notebook: evidence-weighted pairwise chaining with a built-in null

Append-only. Timestamps are biotite local time (PST). Remote root:
`RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111`,
workstream directory `$RD/ws_chain/`.

---

## 2026-11-11 03:15 PST: setup and reading of the two mechanisms

Read `docs/redesign/BRIEF.md` in full, and both cluster preflight files
(`/home/jwestrob/.claude/CLAUDE.md`, `/home/jwestrob/jwestrob/obsidian-vault/biotite-cluster-ops.md`).

Private copies (never edit the originals):
- `$RD/ws_chain/code/production/elsa/` from `$SCALE/production/elsa/`
  (`chain.py` sha256 `c7e40b8b...70d9c9`, identical to the validated snapshot).
- `$RD/ws_chain/code/scale/` from `$SCALE/code/*.py`.

The snapshot has no `neighborhood.py`; the neighbourhood selection that produced the saved
runs is `spacedust_seed_extension.extend_query` (called through
`spacedust_prepared_extension_executor.extend_prepared_queries`). Mechanisms as read:

1. **Selection (`extend_query`).** For query `q`, each of its six neighbours `n` (offsets
   -3..+3) contributes its original seeds `t` (top-60, cosine >= 0.85). Each `t` proposes
   the 3 genes on its strand-consistent side, `c = t +/- 1..3`, on the same reference
   contig. Proposals with cosine(q, c) >= 0.85 are kept. A proposal is "supported"; its
   witness is the proposer with the highest cosine(n, t). The final 60 targets are ranked
   by (supported first, then cosine(q,c) + cosine(n,t), then cosine(q,c)). Every supported
   proposal (q, c) sits at most 3 genes from its witness (n, t) on both axes, so each one
   is by construction a collinear 2-anchor chain with its witness. Seeds have
   cosine >= 0.938 for 99% of pairs in all three models (quantiles below), so the 0.85
   gate passes nearly every proposal, and supported-first ranking evicts unsupported
   originals in favour of these ready-made pairs. This is the chance-block generator of
   BRIEF findings B and C.
2. **Chaining (`chain.py`).** DP over anchors per contig pair and orientation, gap <= 2 on
   both axes; a predecessor is taken if it gives a longer chain, or an equal-length chain
   with higher summed cosine. Endpoints are claimed in order of (length desc, summed
   cosine desc); a path stops at the first claimed anchor and is kept if it has >= 2
   anchors. Blocks are then selected per contig pair greedily by summed cosine with
   paired-span overlap suppression.

Seed cosine quantiles (q01, q10, q50, q90, q99), 66-67 k queries, ~60 seeds each:

| model | seeds | q01 | q10 | q50 | q90 | q99 |
|---|---|---|---|---|---|---|
| esm2_8m | 3,970,270 | 0.938 | 0.970 | 0.986 | 0.992 | 0.996 |
| esm2_650m | 3,971,948 | 0.950 | 0.974 | 0.988 | 0.995 | 0.998 |
| esmc600m | 4,027,458 | 0.912 | 0.947 | 0.973 | 0.987 | 0.995 |

Hold-out split (BRIEF section 9; every third query genome in alphabetical order):
training = Kutzneria sp. 744; Micromonospora sp. ATCC 39149; Streptomyces albus J1074;
Streptomyces ghanaensis ATCC 14672; Streptomyces pristinaespiralis ATCC 25486;
Streptomyces sp. AA4. Held out = **Micromonospora sp. M42** (30 source207 regions),
**Streptomyces himastatinicus ATCC 53653** (0 regions), **Streptomyces sp. C** (0 regions).
Only 30 of the 207 regions sit in held-out genomes, so held-out F1 has a wide interval;
held-out sequence visibility of accepted anchor pairs is the main held-out check.

---

## 2026-11-11 03:40 PST: design and predeclared criteria (written before any run)

### Terms
- **Pair score `s`**: an order-free number for a (query protein, reference protein) pair.
- **Calibrated evidence `w`**: `logit P(sequence-visible | s)`, where sequence-visible means
  MMseqs2 Smith-Waterman E <= 1e-3 against the full reference (BRIEF `aln_seq.tsv` recipe).
- **Chance ratio**: mean block count over the three whole-pipeline permutations divided by
  the observed block count, for the same acceptance rule. Permutation = `ordered_metadata`
  seeds 17/18/19 applied to query gene order before selection and chaining (BRIEF trap 3).

### Step 1. Pair-score choice and calibration map (training genomes only)
Pool sample: 2,500 random eligible queries from training genomes and 1,000 from held-out
genomes (numpy seed 17). For each, dump the full candidate pool of the observed order:
original seeds plus every proposal, with no cosine cut and no 60 cap. Align every unique
(q, c) pair with MMseqs2 (BRIEF recipe). Candidate pair scores:
- `raw`: cosine c.
- `top10`: c minus the mean of the query's ten best seed cosines.
- `zlist`: (c - mean of the query's seed cosines) / their standard deviation.
- `zq`: (c - mu_q) / sd_q, with mu, sd from cosines of q to a fixed panel of 2,048 random
  reference proteins (numpy seed 17), top 1% of panel cosines trimmed.
- `z2`: mean of the query-side and target-side background z (same panel).

Rule: the score with the highest AUROC on the training pairs (seeds and proposals pooled,
unique pairs) is used for that model; a score within 0.005 AUROC of the best is preferred if
it comes earlier in the list above. The map is isotonic regression of the visibility label
on `s`, clipped to P in [0.005, 0.995], so `w` lies in [-5.29, +5.29]. Calibration is
reported on held-out genomes separately for seeds and proposals.

Known circularity: V2-V4 are tuned toward sequence-visible pairs, so their sequence-visible
fraction is partly by construction. The chance ratio and the source207 F1 are the
independent readouts.

### Step 2. Variants (each switchable on its own)
| id | selection | chainer | notes |
|---|---|---|---|
| V0 | published context60 (`extend_query`, unchanged) | published length-first | replay must give 534,990 / 495,161 / 495,380 / 495,342 for esm2_8m |
| V1 | original ANN60 seeds, no selection | published length-first | |
| V2-p50/p80/p95 | evidence-gated: a proposal is kept only if w(witness) >= logit(p) and w(q, c) >= logit(p), no cosine cut; witness = proposer with the highest w; rank = w(q,c) + w(witness) for supported proposals, w(q,c) for unsupported originals; 60 kept | published length-first | p in {0.5, 0.8, 0.95} |
| V3 on {V0, V1, V2-p50, V2-p80, V2-p95} selections | as named | score-first: DP maximizes sum(w) minus gap cost; endpoints claimed by score; truncated paths trimmed to their best-scoring suffix; blocks extracted by score | gap cost none, or concave 0.5*log2(g+1) with g = max gap on either axis; gap <= 2 kept |

V3's built-in chain cut: a chain is kept if its total >= T. **T = 4 is the predeclared
built-in rule** (about two anchors at P = 0.88); T = 0 and the full curve are reported.
Applying T before or after extraction gives the same accepted set, because extraction is
greedy by the same score.

### Step 3. Acceptance rules evaluated for every variant
- **cos5**: per-size cut on mean anchor cosine, sizes 2, 3, 4, 5, 6+, cut by
  `cut_for(..., 0.05)` exactly as in `permuted_panels.py` (comparability with tonight's tables).
- **V4 (q5)**: block statistic = sum of w over the block's anchors; empirical q-value within
  size strata 2, 3, 4, 5, 6+: q(S) = min over t <= S of [(# permuted blocks with stat >= t)/3]
  / (# observed blocks with stat >= t); accept q <= 0.05.
- **raw**: all blocks (V1 and V3 at T = 4 are the candidates for "no post-hoc filter").

### Step 4. Metrics per model x variant x rule
Observed blocks; chance ratio overall and by size; accepted blocks and anchor pairs with
the achieved chance ratio of the accepted set; sequence-visible fraction of accepted anchor
pairs (all genomes, and held-out genomes); source207 mean best-single-block F1, random
placement F1 (200 placements, seed 17), excess with 95% region bootstrap (10,000 draws), and
the paired bootstrap interval of the excess difference against calibrated V0; accepted
blocks by target stratum; wall time of selection and chaining. Models: esm2_8m, esm2_650m,
esmc600m.

### Predeclared criterion
**Gain.** Variant X under rule R is a gain over calibrated V0 (V0 under cos5) for a model if,
with an achieved chance ratio <= 0.05 for its accepted set, it
1. accepts more anchor pairs than calibrated V0,
2. has a sequence-visible fraction of accepted anchor pairs >= calibrated V0's, and
3. has source207 F1 >= calibrated V0's, both observed F1 and F1 excess over random
   placement (point estimates; the paired interval is reported beside them).
A change is recommended for the package when some configuration that contains it is a gain
in at least 2 of the 3 models, and the matched comparison that isolates it (same selection
for a chainer change, same chainer for a selection change) does not lose anchor pairs at
<= 5% chance.

**Ideal.** A raw chance ratio <= 0.05 with no post-hoc filter (V1, or V3 at T = 4).

**Evidence against evidence weighting** (any of these is reported as a negative result):
- no V2 or V3 configuration is a gain in at least 2 of 3 models;
- V1 (no selection) accepts at least as many anchor pairs as the best V2 at <= 5% chance
  with equal or higher visibility and F1: then neighbourhood selection should be dropped,
  not repaired;
- score-first chaining (V3) accepts fewer anchor pairs at <= 5% chance than length-first on
  the same selection, or lowers F1;
- V4 accepts fewer anchor pairs than cos5 on the same blocks: then summed calibrated
  evidence is no better a block statistic than mean cosine.

---

## 2026-11-11 03:27 PST: code and first submissions

Code (local `code/`, mirrored to `$RD/ws_chain/code/`):
- `common.py` paths, `ordered_metadata` and `cut_for` copied verbatim from `permuted_panels.py`,
  pair scores, isotonic evidence map, empirical q-values.
- `chain_v3.py` length-first mode calls the published preprocessing, DP, backtracking and
  paired-overlap kernels of the private snapshot and mirrors `_extract_blocks_flat`;
  score-first mode adds its own DP (`_dp_score_first`) and suffix trimming (`_trim_chains`).
- `extend_v2.py` V2 selection (copy of `extend_query` candidate generation) and `pool_dump`.
- `prep.py` (stage A), `pairs.py` + `align_pairs.sh` (MMseqs2 recipe of `pipeline.sh`),
  `fit_map.py` (stage B/C), `run_variants.py` (D), `summarize.py` (F), `unit_tests.py`.
- Slurm wrappers `run_*.sbatch` (standard_expand, -c 24, exclude list, no mem/time).

Jobs:
- 76290 (array 0-2) `run_prep.sbatch`: stats, pool sample, V0 replay with count asserts and
  the exact-identity check of `chain_v3` length-first against the published chainer.
- 76291 `run_smoke.sbatch` (afterok 76290_0): unit tests, then stages D and F on the first
  3,000 query rows of esm2_8m with a dummy map, output `runs/esm2_8m_smoke`.
- 76292 `run_align_fit.sbatch` (afterok 76290): pool-pair alignment, recipe control
  (20,000 aln_seq pairs re-aligned), map fitting.

---

## 2026-11-11 03:35 PST: correction, V0 replay passed, pool sample sizes

Correction: job 76290 died after the pool stage on a duplicate keyword in the report
dict (`prep.py`, no data affected). Cancelled 76290/76291/76292; resubmitted as
**76301** (prep array), **76304** (smoke, afterok 76301_0), **76305** (align+fit, afterok 76301).

V0 replay (job 76301, published `extend_prepared_queries` + published chainer, private copy):

| model | observed blocks | permuted seeds 17 / 18 / 19 | saved | extension s | chain s |
|---|---|---|---|---|---|
| esm2_8m | 534,990 (1,482,427 anchor pairs) | 495,161 / 495,380 / 495,342 | identical | 41.6 | 32.4 |

`chain_v3` length-first reproduces the published observed blocks exactly (all span, size,
orientation, sorted anchor position lists and rounded chain scores) for esm2_8m, esm2_650m
and esmc600m. Unit tests (job 76304) pass: score-first chains bridge-skip a negative anchor
(9 anchors, score 18.0; 17.5 with the concave gap), length-first keeps the bridge
(10 anchors), q-values as computed by hand.

Background statistics, esm2_8m: query background mean cosine to the 2,048-protein panel
median 0.799 (1st-99th pct 0.575-0.866), sd median 0.078; reference the same within 0.03.
A typical seed (cosine 0.986) is therefore about 2.4 background sd above the query's mean.

Pool sample (3,500 queries per model; unique pairs = seeds + proposals, no cutoff):

| model | unique pairs | seeds | proposals | witness occurrences |
|---|---|---|---|---|
| esm2_8m | 2,132,575 | 209,837 | 1,922,738 | 2,028,486 |
| esm2_650m | 2,037,889 | 209,947 | 1,827,942 | 2,027,431 |
| esmc600m | 2,104,356 | 209,942 | 1,894,414 | 2,052,360 |

---

## 2026-11-11 03:45 PST: step 1 results (pair score and calibration map), job 76305

Alignment: 5,869,053 unique pairs (5,849,898 pool + 20,000 control) aligned in 43 s with the
`pipeline.sh` recipe (`align/pool/aln.tsv`). Recipe control: all 20,000 re-aligned
`aln_seq.tsv` pairs give identical E-values and bit scores (`align/pool/control_check.json`).
Every pool pair has a label (0 unscored).

Sequence-visible prevalence in the candidate pool (training genomes; held-out within 0.03):

| model | seeds (top-60 ANN) | proposals (all, no cutoff) | proposals passing the published cosine >= 0.85 gate |
|---|---|---|---|
| esm2_8m | 0.474 | 0.023 | 44% pass, 0.050 visible |
| esm2_650m | 0.614 | 0.023 | 93% pass, 0.024 visible |
| esmc600m | 0.392 | 0.019 | 44% pass, 0.040 visible |

The published gate admits 44-93% of all proposals while only 2-5% of them are
sequence-visible: this is the measured form of BRIEF finding C at the pair level.

Training AUROC (pooled unique pairs; seeds-only in parentheses):

| model | raw | top10 | zlist | zq | z2 | chosen |
|---|---|---|---|---|---|---|
| esm2_8m | 0.962 (0.685) | 0.956 (0.541) | 0.953 (0.572) | 0.956 (0.726) | **0.968** (0.737) | z2 |
| esm2_650m | **0.970** (0.780) | 0.964 (0.606) | 0.954 (0.594) | 0.964 (0.550) | 0.969 (0.593) | raw (z2 within 0.005 but later in the list) |
| esmc600m | **0.941** (0.778) | 0.932 (0.588) | 0.936 (0.639) | 0.869 (0.265) | 0.884 (0.288) | raw |

Held-out AUROC of the chosen score: 0.968 / 0.970 / 0.942 pooled. The within-list
normalizations (`top10`, `zlist`) are worse than raw cosine in every model, and background
z-scores help only ESM2-8M; for ESM-C 600M they invert the ordering among seeds (AUROC 0.27).

Calibration on held-out genomes (isotonic, 358-410 knots): pooled Brier 0.033 / 0.030 / 0.031
against 0.063 / 0.075 / 0.052 for the prevalence-only forecast. Seeds are under-predicted
at the low end in esm2_8m (lowest decile predicted 0.12, observed 0.28) and proposals
slightly over-predicted at the top (0.26 vs 0.21); esm2_650m and esmc600m seeds are within
0.05 except the lowest decile (0.10 vs 0.22; 0.05 vs 0.08). This is a stated limitation of the
pooled map: at a given score a top-60 seed is more often a homolog than a proposal.

V2 gates on proposals (training genomes; own gate / own and witness gate; visible fraction):

| model | p | pass own | visible | pass both | visible |
|---|---|---|---|---|---|
| esm2_8m | 0.5 | 19,282 (1.4%) | 0.73 | 15,348 | 0.74 |
| esm2_8m | 0.8 | 7,702 | 0.77 | 4,380 | 0.78 |
| esm2_8m | 0.95 | 844 | 0.97 | 173 | 1.00 |
| esm2_650m | 0.5 | 12,320 (0.9%) | 0.91 | 10,919 | 0.92 |
| esm2_650m | 0.8 | 4,358 | 1.00 | 3,128 | 1.00 |
| esm2_650m | 0.95 | 1,327 | 1.00 | 723 | 1.00 |
| esmc600m | 0.5 | 1,878 (0.1%) | 0.91 | 1,381 | 0.93 |
| esmc600m | 0.8 | 438 | 0.98 | 230 | 0.99 |
| esmc600m | 0.95 | 89 | 1.00 | 30 | 1.00 |

Held-out genomes agree except esmc600m at p >= 0.8 (27 and 0 proposals pass). Expectation
before stage D: V2 at p >= 0.8 is close to V1 for ESM-C 600M.

Smoke test (job 76304, first 3,000 query rows, dummy map) ran stages D and F end to end.

---

## 2026-11-11 04:05 PST: stage D raw output (job 76335), no acceptance rule applied

Job 76335 (array 0-2) ran 5 selections x 3 chainers x 4 orders per model in 17-21 min on
24 CPUs. The context60 length-first counts again equal the saved ones for every order.
Job 76336 collected 3,347,859 distinct block anchor pairs over all variants and models;
108,232 lacked a label and were aligned (9 s). Job 76337 (summaries) started 04:01.

Raw blocks (observed), anchor pairs, mean permuted blocks, chance ratio, and wall time of
the observed run (selection s / chaining s, 24 CPUs):

| model | variant | blocks | pairs | permuted | ratio | sel s | chain s |
|---|---|---|---|---|---|---|---|
| esm2_8m | V0 context60 + lf | 534,990 | 1,482,427 | 495,294 | 0.926 | 40 | 19 |
| esm2_8m | V0 sel + V3 sf0 | 251,903 | 558,477 | 199,041 | 0.790 | 40 | 11 |
| esm2_8m | V0 sel + V3 sfc | 236,215 | 515,066 | 189,328 | 0.802 | 40 | 9 |
| esm2_8m | V1 seeds + lf | 126,775 | 344,086 | 1,426 | 0.011 | 0 | 10 |
| esm2_8m | V1 + sf0 | 71,511 | 162,235 | 614 | 0.009 | 0 | 9 |
| esm2_8m | V1 + sfc | 64,696 | 143,334 | 418 | 0.006 | 0 | 9 |
| esm2_8m | V2 p50 + lf | 205,562 | 538,739 | 23,455 | 0.114 | 22 | 11 |
| esm2_8m | V2 p50 + sf0 | 154,526 | 357,551 | 22,656 | 0.147 | 22 | 9 |
| esm2_8m | V2 p50 + sfc | 138,735 | 316,507 | 15,638 | 0.113 | 22 | 9 |
| esm2_8m | V2 p80 + lf | 145,425 | 390,181 | 3,447 | 0.024 | 20 | 10 |
| esm2_8m | V2 p80 + sf0 | 90,764 | 208,371 | 2,635 | 0.029 | 20 | 9 |
| esm2_8m | V2 p80 + sfc | 84,231 | 189,861 | 2,438 | 0.029 | 20 | 9 |
| esm2_8m | V2 p95 + lf | 129,198 | 349,227 | 1,442 | 0.011 | 19 | 10 |
| esm2_8m | V2 p95 + sf0 | 73,963 | 167,388 | 629 | 0.009 | 19 | 9 |
| esm2_8m | V2 p95 + sfc | 67,160 | 148,533 | 433 | 0.006 | 19 | 9 |
| esm2_650m | V0 context60 + lf | 478,610 | 1,513,349 | 318,702 | 0.666 | 62 | 19 |
| esm2_650m | V0 sel + V3 sf0 | 232,715 | 686,930 | 3,598 | 0.015 | 62 | 11 |
| esm2_650m | V0 sel + V3 sfc | 221,856 | 659,328 | 2,558 | 0.012 | 62 | 9 |
| esm2_650m | V1 seeds + lf | 225,862 | 642,321 | 2,098 | 0.009 | 0 | 11 |
| esm2_650m | V1 + sf0 | 201,979 | 560,327 | 1,437 | 0.007 | 0 | 10 |
| esm2_650m | V1 + sfc | 193,972 | 539,586 | 1,084 | 0.006 | 0 | 10 |
| esm2_650m | V2 p50 + lf | 272,157 | 794,099 | 6,493 | 0.024 | 42 | 11 |
| esm2_650m | V2 p50 + sf0 | 250,904 | 713,069 | 5,835 | 0.023 | 42 | 11 |
| esm2_650m | V2 p50 + sfc | 238,375 | 679,519 | 3,892 | 0.016 | 42 | 10 |
| esm2_650m | V2 p80 + lf | 235,426 | 683,759 | 2,330 | 0.010 | 28 | 11 |
| esm2_650m | V2 p80 + sf0 | 212,231 | 601,982 | 1,669 | 0.008 | 28 | 10 |
| esm2_650m | V2 p80 + sfc | 204,602 | 582,239 | 1,317 | 0.006 | 28 | 10 |
| esm2_650m | V2 p95 + lf | 226,744 | 650,586 | 2,116 | 0.009 | 25 | 11 |
| esm2_650m | V2 p95 + sf0 | 202,985 | 568,596 | 1,455 | 0.007 | 25 | 10 |
| esm2_650m | V2 p95 + sfc | 195,069 | 548,058 | 1,102 | 0.006 | 25 | 10 |
| esmc600m | V0 context60 + lf | 463,807 | 1,377,536 | 368,580 | 0.795 | 50 | 18 |
| esmc600m | V0 sel + V3 sf0 | 146,819 | 390,203 | 31,052 | 0.211 | 50 | 11 |
| esmc600m | V0 sel + V3 sfc | 142,168 | 372,005 | 24,752 | 0.174 | 50 | 9 |
| esmc600m | V1 seeds + lf | 168,577 | 482,221 | 3,161 | 0.019 | 0 | 10 |
| esmc600m | V1 + sf0 | 120,225 | 321,853 | 831 | 0.007 | 0 | 9 |
| esmc600m | V1 + sfc | 114,660 | 303,977 | 571 | 0.005 | 0 | 9 |
| esmc600m | V2 p50 + lf | 172,104 | 499,338 | 3,369 | 0.020 | 27 | 11 |
| esmc600m | V2 p50 + sf0 | 124,806 | 339,403 | 1,041 | 0.008 | 27 | 10 |
| esmc600m | V2 p50 + sfc | 118,704 | 320,047 | 703 | 0.006 | 27 | 10 |
| esmc600m | V2 p80 + lf | 168,512 | 484,270 | 3,162 | 0.019 | 24 | 11 |
| esmc600m | V2 p95 + lf | 168,601 | 482,339 | 3,161 | 0.019 | 23 | 11 |

(esmc600m V2 p80/p95 with score-first are within 0.3% of V1 + sf0/sfc; full rows in
`results/ablation_table.tsv`.)

First reading, before visibility and F1: the raw chance ratio falls from 0.67-0.93 (V0) to
0.006-0.03 whenever the binary support gate is removed (V1) or replaced by evidence at
p >= 0.8 (V2), with no post-hoc filter. Score-first chaining alone fixes V0's selection only
for esm2_650m (0.015), whose raw-cosine map gives most chance proposals negative weight; it
leaves esm2_8m at 0.79 and esmc600m at 0.21 at T = 0. V1's ANN seeds give 126,775 blocks for
esm2_8m (BRIEF's 116,931 is the exact global top-60 arm of `full_reference_interventions`,
a different seed set).

---

## 2026-11-11 04:35 PST: predeclared test outcome (jobs 76337, 76378), and one post hoc follow-up

Calibrated V0 reproduces tonight's `permuted_panels` numbers exactly (esm2_8m 111,389
blocks, visible 0.8416, F1 0.6519, random 0.6580; esm2_650m 265,639 / 0.7213 / 0.7544;
esmc600m 147,858 / F1 0.7624 / excess +0.048). **No variant x rule meets the predeclared gain
criterion in any model** (`gain` column of `results/ablation_table.tsv` is False for all 195
rows). Full tables: `results/ABLATION.md`. Details are in REPORT.md; the decisive numbers:

| model | calibrated V0 pairs (visible pairs) | best repair at <= 5% chance by pairs | its pairs (visible pairs) | F1 excess V0 / repair, paired dEx [95%] |
|---|---|---|---|---|
| esm2_8m | 536,198 (451,258) | V2-p50 + lf, cos5 | 411,875 (395,960) | -0.006 / +0.002, +0.008 [-0.023, +0.040] |
| esm2_650m | 1,052,580 (759,235) | V2-p50 + lf, raw (ratio 0.024) | 794,099 (744,829) | -0.005 / -0.006, -0.001 [-0.021, +0.020] |
| esmc600m | 683,194 (535,335) | V2-p50 + lf, raw (ratio 0.020) | 499,338 (477,290) | +0.048 / +0.022, -0.026 [-0.049, -0.003] |

Diagnostic (`diag.py`, job 76378, post hoc, descriptive): of calibrated V0's accepted pairs,
supported proposals are 46% / 43% / 38% (8M / 650M / 600M) with visibility 0.68 / 0.42 / 0.48;
original seeds are 0.98 / 0.95 / 0.97 visible. The pair-level map gives negative weight
(P < 0.5) to 75% / 75% / 93% of these accepted proposals, of which 58% / 25% / 44% are
sequence-visible, and to 55% / 13% / 32% of the accepted original seeds, of which 97% / 85% / 93%
are visible. In V1 blocks of esm2_8m, 190,741 of 344,086 anchors have w < 0 and 96% of them
are visible. V2-p50 misses 158,458 / 125,965 / 117,213 of V0's visible accepted pairs.

Root cause, as read from these numbers: the weight w = logit P(visible | s) is calibrated
on the candidate pool, whose prevalence is 0.067 / 0.084 / 0.057, so every anchor carries the
pool prior logit(pi) of about -2.6. A chain of k anchors pays that prior k times, while a
block needs it once; collinearity itself is the evidence that lifts weak pairs, and the
pair-level map cannot see it. The same pair-level map gates V2's proposals before any
collinearity is known.

**Post hoc follow-up (exploratory; outside the predeclared test, labelled as such in the
report).** Weight = log-likelihood ratio w' = logit P - logit(pi) (prior removed, so a typical
seed is positive and a random adjacent gene negative); V2 adds a gate at P >= pi
(`ev_pprior`, i.e. w' >= 0 for both witness and proposal) and ranks by summed w'. Selections:
context60, seeds, ev_p50, ev_pprior; chainers lf, sf0, sfc; same permutations, rules and
metrics; V4 statistic = sum of w'. Reading fixed now: the prior double-counting explains the
V3 loss if V1 + V3(w') accepts at least 95% of V1 + lf's anchor pairs at <= 5% chance; the
follow-up would justify a further predeclared test (not a package change) if some w'
configuration reaches calibrated V0's anchor pairs at <= 5% chance with visibility >= V0's.

---

## 2026-11-11 05:15 PST: post hoc LLR follow-up results (jobs 76381, 76382, 76383)

76381 (variants with w' = logit P - logit(pi), pi = 0.067 / 0.084 / 0.057, offsets -2.63 /
-2.39 / -2.82), 76382 (1,269,883 new pairs aligned; 5,118,825 distinct block pairs in all
runs), 76383 (summaries). Tables: `results/ablation_table_llr.tsv`,
`results/<model>_llr/table.tsv`.

**Predeclared reading met: per-anchor prior double-counting explains the V3 loss.**
V1 + score-first with w' keeps 341,654 / 633,364 / 476,946 anchor pairs, i.e. 99.3% / 98.6% /
98.9% of V1 + length-first (posterior weights kept 47% / 87% / 67%), at the same chance ratio
(0.011 / 0.009 / 0.018).

**Trigger for a further test not met as written:** no w' configuration reaches calibrated V0's
anchor pairs at <= 5% chance. Closest, all with higher visibility than V0:

| model | configuration (post hoc) | blocks | pairs (% of V0 cos5) | visible pairs (% of V0) | chance | visible | F1 / random | excess [95%] | dEx vs V0 cos5 [95%] |
|---|---|---|---|---|---|---|---|---|---|
| esm2_8m | V0 sel + V3(w') no gap, cos5 | 120,547 | 519,133 (97%) | 461,794 (102%) | 0.030 | 0.890 | 0.634 / 0.636 | -0.002 [-0.035, +0.032] | +0.004 [-0.015, +0.022] |
| esm2_650m | V0 sel + V3(w') no gap, cos5 | 262,686 | 864,268 (82%) | 740,676 (98%) | 0.027 | 0.857 | 0.670 / 0.678 | -0.008 [-0.039, +0.023] | -0.003 [-0.021, +0.015] |
| esm2_650m | V0 sel + V3(w') no gap, built-in T4 (no post-hoc filter) | 230,788 | 798,531 (76%) | 712,375 (94%) | 0.010 | 0.892 | 0.651 / 0.657 | -0.006 [-0.039, +0.026] | -0.001 [-0.022, +0.018] |
| esmc600m | V0 sel + V3(w') no gap, cos5 | 153,798 | 594,642 (87%) | 524,223 (98%) | 0.031 | 0.882 | 0.732 / 0.663 | +0.069 [+0.041, +0.095] | **+0.021 [+0.006, +0.035]** |
| esmc600m | V0 sel + V3(w') concave, cos5 | 153,348 | 579,168 (85%) | 523,861 (98%) | 0.028 | 0.905 | 0.710 / 0.641 | +0.069 [+0.041, +0.096] | +0.022 [+0.006, +0.037] |
| esmc600m | ev_pprior + V3(w') concave, q5 | 103,558 | 464,846 (68%) | 421,775 (79%) | 0.028 | 0.907 | 0.702 / 0.618 | +0.085 [+0.055, +0.114] | +0.037 [+0.012, +0.062] |

Other post hoc rows: a V2 gate at the pool prior (`ev_pprior`) is nearly as permissive as the
published gate (raw chance 0.968 / 0.583 / 0.820); the built-in T4 cut with w' holds chance at
<= 0.05 only for esm2_650m (0.010; esm2_8m 0.593, esmc600m 0.085 on the V0 selection).

Reading: with prior-free weights, score-first chaining on the published selection is the most
promising repair found. Under the per-size cosine cut it keeps 82-97% of calibrated V0's pairs,
98-102% of its sequence-visible pairs, raises the visible fraction by 0.05-0.14, and for
ESM-C 600M raises F1 excess by +0.021 [+0.006, +0.035]. These configurations were chosen
after the predeclared test, from about 45 rows per model, so the ESM-C interval is not
corrected for that selection; the result is a candidate for a new predeclared test with
held-out query genomes, not a package change.

---

## 2026-11-11 05:20 PST: closing report

The harness refused to create `REPORT.md` from this agent ("subagents should return findings
as text"); this entry is the report, for the coordinator to promote to `REPORT.md`.

### Findings
1. **The binary support gate generates the chance blocks; removing it fixes the raw null with
   no filter.** Raw permuted/observed: V0 0.926 / 0.666 / 0.795 (esm2_8m / esm2_650m /
   esmc600m); V1 (seeds only) 0.011 / 0.009 / 0.019; V2-p80 0.024 / 0.010 / 0.019. The
   predeclared ideal (raw <= 0.05, no post-hoc filter) is met by V1 and V2-p80 in all three
   models. The published cosine >= 0.85 gate admits 44-93% of all proposals, 2-5% of which are
   sequence-visible.
2. **No predeclared variant is a gain over calibrated V0 in any model** (195 rows, `gain`
   False throughout). Calibrated V0 accepts the most anchor pairs (536,198 / 1,052,580 /
   683,194) and the most sequence-visible pairs (451,258 / 759,235 / 535,335). The best
   predeclared repair, V2-p50, reaches 77% / 75% / 73% of V0's pairs and 88% / 98% / 89% of its
   visible pairs at visible fraction 0.961 / 0.938 / 0.956 (V0 0.842 / 0.721 / 0.784).
3. **F1.** ESM2: every variant at <= 5% chance is within the +/-0.04 region-bootstrap
   resolution of calibrated V0 (excess -0.006 / -0.005). ESM-C 600M: calibrated V0 keeps
   +0.048 [+0.023, +0.071]; every predeclared repair is lower by 0.023-0.10, intervals
   excluding 0 for V1, V2 and V3 at T = 4.
4. **V3 as specified (posterior log-odds weights) loses pairs**: score-first keeps 47% / 87% /
   67% of length-first's raw pairs on V1's selection; built-in T = 4 keeps 8% / 59% / 26%.
   Concave gap cost moves accepted pairs by 3-11% and never reverses a comparison.
5. **V4 is no better than the per-size cosine cut**: on V0's blocks 447,193 vs 536,198 pairs
   (8M), 1,064,403 vs 1,052,580 (650M), 647,020 vs 683,194 (600M), equal visibility.
6. **Root cause (diagnostic):** 38-46% of calibrated V0's accepted pairs are supported
   proposals (syntenic targets beyond the top-60 budget), 42-68% sequence-visible; the pair-level
   map gives 75-93% of them, and 13-55% of accepted seeds (85-97% visible), negative weight,
   because w = logit P carries the pool prior (logit 0.06-0.08, about -2.6) once per anchor.
   Removing the prior (post hoc, w' = LLR) restores 99% of V1's pairs under score-first chaining.
7. **Pair scores.** Raw cosine is best for esm2_650m and esmc600m (pooled AUROC 0.970 / 0.941),
   two-sided background z for esm2_8m (0.968 vs 0.962). Within-list normalizations (minus top-10
   mean; z within the seed list) are worse than raw cosine in all three models.

### Ablation table (predeclared; key rows)
chance = mean permuted / observed blocks of the accepted set; visible = sequence-visible
fraction of accepted pairs (held-out genomes in parentheses); dEx = F1 excess minus calibrated
V0's, paired 95% region bootstrap. Every row: `results/ABLATION.md`.

#### esm2_8m

| variant | rule | blocks | anchor pairs | visible pairs | chance ratio | visible frac (held-out) | F1 | F1 random | F1 excess [95%] | dEx vs V0 cos5 [95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | raw | 534,990 | 1,482,427 | 785,315 | 0.926 | 0.530 (0.527) | 0.757 | 0.782 | -0.025 [-0.042, -0.008] | -0.019 [-0.045, +0.009] |
| V0 | cos5 | 111,389 | 536,198 | 451,258 | 0.034 | 0.842 (0.843) | 0.652 | 0.658 | -0.006 [-0.040, +0.028] | +0.000 [+0.000, +0.000] |
| V0 | q5 | 79,892 | 447,193 | 374,465 | 0.028 | 0.837 (0.837) | 0.628 | 0.628 | +0.001 [-0.035, +0.037] | +0.007 [-0.020, +0.034] |
| V1 | raw | 126,775 | 344,086 | 333,309 | 0.011 | 0.969 (0.968) | 0.516 | 0.535 | -0.018 [-0.052, +0.016] | -0.012 [-0.042, +0.017] |
| V2-p50 | cos5 | 142,130 | 411,875 | 395,960 | 0.030 | 0.961 (0.962) | 0.556 | 0.553 | +0.002 [-0.033, +0.038] | +0.008 [-0.023, +0.040] |
| V2-p50 | q5 | 59,644 | 246,903 | 240,170 | 0.003 | 0.973 (0.973) | 0.485 | 0.480 | +0.005 [-0.038, +0.048] | +0.011 [-0.028, +0.049] |
| V2-p80 | raw | 145,425 | 390,181 | 374,702 | 0.024 | 0.960 (0.960) | 0.535 | 0.542 | -0.007 [-0.040, +0.026] | -0.001 [-0.028, +0.026] |
| V2-p95 | raw | 129,198 | 349,227 | 338,491 | 0.011 | 0.969 (0.969) | 0.516 | 0.535 | -0.019 [-0.053, +0.016] | -0.012 [-0.042, +0.017] |
| V0 sel + V3 (no gap) | T0 | 142,737 | 340,145 | 287,794 | 0.363 | 0.846 (0.843) | 0.526 | 0.477 | +0.049 [+0.016, +0.081] | +0.055 [+0.016, +0.094] |
| V0 sel + V3 (no gap) | T4 | 23,080 | 82,879 | 79,872 | 0.033 | 0.964 (0.963) | 0.190 | 0.212 | -0.022 [-0.061, +0.019] | -0.016 [-0.059, +0.027] |
| V0 sel + V3 (no gap) | q5 | 32,871 | 120,413 | 116,651 | 0.007 | 0.969 (0.970) | 0.323 | 0.308 | +0.015 [-0.030, +0.059] | +0.021 [-0.024, +0.064] |
| V0 sel + V3 (concave) | T4 | 19,844 | 67,000 | 64,664 | 0.021 | 0.965 (0.964) | 0.154 | 0.149 | +0.005 [-0.034, +0.045] | +0.011 [-0.031, +0.054] |
| V1 + V3 (no gap) | T0 | 56,747 | 132,707 | 129,576 | 0.008 | 0.976 (0.977) | 0.363 | 0.356 | +0.007 [-0.030, +0.044] | +0.013 [-0.027, +0.053] |
| V1 + V3 (no gap) | T4 | 8,275 | 28,761 | 28,468 | 0.002 | 0.990 (0.989) | 0.129 | 0.165 | -0.037 [-0.072, +0.001] | -0.030 [-0.072, +0.012] |
| V2-p50 + V3 (no gap) | T4 | 22,596 | 76,652 | 72,670 | 0.054 | 0.948 (0.945) | 0.196 | 0.205 | -0.009 [-0.050, +0.033] | -0.003 [-0.049, +0.042] |
| V2-p50 + V3 (concave) | cos5 | 88,871 | 216,779 | 207,354 | 0.037 | 0.957 (0.957) | 0.416 | 0.375 | +0.041 [+0.004, +0.079] | +0.047 [+0.007, +0.086] |
| V2-p80 + V3 (no gap) | T4 | 17,992 | 56,114 | 53,833 | 0.020 | 0.959 (0.959) | 0.146 | 0.176 | -0.030 [-0.066, +0.008] | -0.024 [-0.066, +0.018] |

#### esm2_650m

| variant | rule | blocks | anchor pairs | visible pairs | chance ratio | visible frac (held-out) | F1 | F1 random | F1 excess [95%] | dEx vs V0 cos5 [95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | raw | 478,610 | 1,513,349 | 812,458 | 0.666 | 0.537 (0.533) | 0.788 | 0.799 | -0.011 [-0.028, +0.005] | -0.006 [-0.019, +0.007] |
| V0 | cos5 | 265,639 | 1,052,580 | 759,235 | 0.038 | 0.721 (0.723) | 0.754 | 0.759 | -0.005 [-0.028, +0.017] | +0.000 [+0.000, +0.000] |
| V0 | q5 | 268,783 | 1,064,403 | 765,173 | 0.039 | 0.719 (0.721) | 0.754 | 0.762 | -0.008 [-0.030, +0.014] | -0.003 [-0.004, -0.001] |
| V1 | raw | 225,862 | 642,321 | 607,966 | 0.009 | 0.947 (0.947) | 0.578 | 0.582 | -0.004 [-0.038, +0.029] | +0.001 [-0.022, +0.023] |
| V2-p50 | cos5 | 272,157 | 794,099 | 744,829 | 0.024 | 0.938 (0.941) | 0.602 | 0.608 | -0.006 [-0.037, +0.025] | -0.001 [-0.021, +0.020] |
| V2-p50 | q5 | 272,157 | 794,099 | 744,829 | 0.024 | 0.938 (0.941) | 0.602 | 0.608 | -0.006 [-0.037, +0.025] | -0.001 [-0.021, +0.020] |
| V2-p80 | raw | 235,426 | 683,759 | 649,235 | 0.010 | 0.950 (0.950) | 0.581 | 0.585 | -0.004 [-0.037, +0.029] | +0.001 [-0.021, +0.023] |
| V2-p95 | raw | 226,744 | 650,586 | 616,251 | 0.009 | 0.947 (0.948) | 0.578 | 0.582 | -0.004 [-0.038, +0.028] | +0.001 [-0.022, +0.022] |
| V0 sel + V3 (no gap) | T0 | 203,204 | 627,908 | 598,363 | 0.008 | 0.953 (0.956) | 0.573 | 0.581 | -0.008 [-0.044, +0.027] | -0.003 [-0.027, +0.020] |
| V0 sel + V3 (no gap) | T4 | 114,190 | 433,946 | 424,986 | 0.003 | 0.979 (0.981) | 0.507 | 0.531 | -0.025 [-0.067, +0.019] | -0.020 [-0.054, +0.014] |
| V0 sel + V3 (no gap) | q5 | 232,715 | 686,930 | 631,609 | 0.015 | 0.919 (0.922) | 0.594 | 0.608 | -0.013 [-0.046, +0.020] | -0.008 [-0.029, +0.012] |
| V0 sel + V3 (concave) | T4 | 108,991 | 416,595 | 409,312 | 0.003 | 0.983 (0.984) | 0.481 | 0.518 | -0.037 [-0.080, +0.006] | -0.032 [-0.067, +0.001] |
| V1 + V3 (no gap) | T0 | 196,371 | 549,111 | 528,296 | 0.007 | 0.962 (0.964) | 0.553 | 0.565 | -0.012 [-0.047, +0.022] | -0.007 [-0.031, +0.016] |
| V1 + V3 (no gap) | T4 | 114,677 | 376,852 | 371,650 | 0.003 | 0.986 (0.987) | 0.492 | 0.523 | -0.031 [-0.072, +0.011] | -0.026 [-0.060, +0.006] |
| V2-p50 + V3 (no gap) | T4 | 131,178 | 458,457 | 451,301 | 0.009 | 0.984 (0.986) | 0.505 | 0.534 | -0.029 [-0.070, +0.012] | -0.024 [-0.057, +0.007] |
| V2-p50 + V3 (concave) | cos5 | 238,375 | 679,519 | 648,925 | 0.016 | 0.955 (0.958) | 0.563 | 0.577 | -0.014 [-0.048, +0.020] | -0.009 [-0.033, +0.014] |
| V2-p80 + V3 (no gap) | T4 | 123,318 | 415,370 | 410,044 | 0.004 | 0.987 (0.988) | 0.493 | 0.525 | -0.032 [-0.073, +0.010] | -0.027 [-0.060, +0.006] |

#### esmc600m

| variant | rule | blocks | anchor pairs | visible pairs | chance ratio | visible frac (held-out) | F1 | F1 random | F1 excess [95%] | dEx vs V0 cos5 [95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| V0 | raw | 463,807 | 1,377,536 | 667,088 | 0.795 | 0.484 (0.485) | 0.818 | 0.793 | +0.025 [+0.011, +0.039] | -0.022 [-0.040, -0.002] |
| V0 | cos5 | 147,858 | 683,194 | 535,335 | 0.034 | 0.784 (0.788) | 0.762 | 0.715 | +0.048 [+0.023, +0.071] | +0.000 [+0.000, +0.000] |
| V0 | q5 | 129,881 | 647,020 | 503,961 | 0.033 | 0.779 (0.782) | 0.747 | 0.708 | +0.039 [+0.011, +0.065] | -0.009 [-0.022, +0.003] |
| V1 | raw | 168,577 | 482,221 | 461,563 | 0.019 | 0.957 (0.959) | 0.595 | 0.575 | +0.020 [-0.012, +0.052] | -0.028 [-0.051, -0.004] |
| V2-p50 | cos5 | 172,104 | 499,338 | 477,290 | 0.020 | 0.956 (0.958) | 0.600 | 0.578 | +0.022 [-0.009, +0.054] | -0.026 [-0.049, -0.003] |
| V2-p50 | q5 | 172,104 | 499,338 | 477,290 | 0.020 | 0.956 (0.958) | 0.600 | 0.578 | +0.022 [-0.009, +0.054] | -0.026 [-0.049, -0.003] |
| V2-p80 | raw | 168,512 | 484,270 | 463,601 | 0.019 | 0.957 (0.959) | 0.595 | 0.575 | +0.020 [-0.012, +0.052] | -0.028 [-0.051, -0.004] |
| V2-p95 | raw | 168,601 | 482,339 | 461,682 | 0.019 | 0.957 (0.959) | 0.595 | 0.575 | +0.020 [-0.012, +0.052] | -0.028 [-0.051, -0.004] |
| V0 sel + V3 (no gap) | T0 | 109,284 | 315,133 | 309,768 | 0.022 | 0.983 (0.986) | 0.536 | 0.518 | +0.018 [-0.018, +0.055] | -0.029 [-0.057, -0.002] |
| V0 sel + V3 (no gap) | T4 | 29,419 | 136,551 | 135,603 | 0.002 | 0.993 (0.994) | 0.374 | 0.419 | -0.045 [-0.091, +0.003] | -0.092 [-0.134, -0.050] |
| V0 sel + V3 (no gap) | q5 | 121,107 | 338,779 | 329,673 | 0.036 | 0.973 (0.977) | 0.559 | 0.534 | +0.025 [-0.009, +0.059] | -0.023 [-0.047, +0.002] |
| V0 sel + V3 (concave) | T4 | 26,010 | 121,277 | 120,664 | 0.002 | 0.995 (0.995) | 0.346 | 0.400 | -0.054 [-0.100, -0.006] | -0.101 [-0.144, -0.059] |
| V1 + V3 (no gap) | T0 | 108,179 | 297,761 | 295,347 | 0.005 | 0.992 (0.993) | 0.525 | 0.508 | +0.017 [-0.019, +0.054] | -0.030 [-0.060, -0.001] |
| V1 + V3 (no gap) | T4 | 30,097 | 125,759 | 125,622 | 0.002 | 0.999 (0.999) | 0.373 | 0.421 | -0.048 [-0.093, -0.001] | -0.095 [-0.136, -0.053] |
| V2-p50 + V3 (no gap) | T4 | 31,286 | 134,565 | 134,399 | 0.002 | 0.999 (0.999) | 0.374 | 0.422 | -0.048 [-0.093, -0.001] | -0.095 [-0.136, -0.054] |
| V2-p50 + V3 (concave) | cos5 | 118,704 | 320,047 | 316,031 | 0.006 | 0.987 (0.989) | 0.514 | 0.506 | +0.009 [-0.026, +0.044] | -0.039 [-0.068, -0.011] |
| V2-p80 + V3 (no gap) | T4 | 30,425 | 128,372 | 128,235 | 0.002 | 0.999 (0.999) | 0.373 | 0.421 | -0.048 [-0.093, -0.001] | -0.095 [-0.136, -0.054] |

### Accepted blocks by target stratum (cos5; blocks, sequence-visible pair fraction)

| model | variant | Streptomyces | same class | same phylum | other phylum | no GTDB |
|---|---|---|---|---|---|---|
| esm2_8m | V0 | 2,937 (0.88) | 101,005 (0.84) | 2,661 (0.78) | 4,474 (0.74) | 312 (0.80) |
| esm2_8m | V1 | 3,693 (0.99) | 114,656 (0.97) | 3,280 (0.93) | 4,798 (0.89) | 348 (0.96) |
| esm2_8m | V2-p50 | 3,871 (0.98) | 125,914 (0.96) | 4,085 (0.92) | 7,856 (0.90) | 404 (0.94) |
| esm2_8m | V2-p80 | 3,995 (0.98) | 127,234 (0.96) | 4,041 (0.93) | 9,748 (0.90) | 407 (0.96) |
| esm2_650m | V0 | 5,402 (0.78) | 233,800 (0.72) | 8,437 (0.66) | 17,075 (0.69) | 925 (0.65) |
| esm2_650m | V1 | 5,153 (0.97) | 202,608 (0.95) | 5,985 (0.93) | 11,408 (0.88) | 708 (0.90) |
| esm2_650m | V2-p50 | 5,871 (0.96) | 240,491 (0.94) | 8,402 (0.92) | 16,471 (0.86) | 922 (0.89) |
| esmc600m | V0 | 3,829 (0.82) | 136,920 (0.79) | 2,871 (0.72) | 3,840 (0.68) | 398 (0.74) |
| esmc600m | V1 | 5,198 (0.92) | 156,053 (0.96) | 3,114 (0.94) | 3,744 (0.91) | 468 (0.90) |
| esmc600m | V2-p50 | 5,253 (0.92) | 159,029 (0.96) | 3,238 (0.94) | 4,098 (0.87) | 486 (0.90) |

By blocks, V2 accepts more than calibrated V0 in every stratum of esm2_8m (cross-phylum 9,748 vs
4,474 at p80) and esmc600m; V0 accepts more anchor pairs because its blocks are longer.

### Wall time (24 CPUs, observed run; each permutation costs the same)
Selection: V0 40-62 s, V2 19-42 s, V1 0 s. Chaining: length-first 10-19 s, score-first 9-11 s.
A built-in null of three whole-pipeline permutations adds about 4 min per model for V0 and 1 min
for V1; whole stage D (15 variants x 4 orders) took 17-21 min per model.

### What to carry into the package
| change | verdict | basis |
|---|---|---|
| Built-in permutation null (3 whole-pipeline permutations) + per-size mean-cosine cut | **carry** | strongest acceptance rule tested; minutes per model; it is what makes the published selection usable |
| Remove the binary support gate (V1, or V2 at p >= 0.8) | carry as an option, not the default | raw chance 0.01-0.02 with no filter and visibility 0.94-0.97, but 23-30% fewer anchor pairs and lower ESM-C F1 than calibrated V0 |
| V2 evidence-gated selection with a pair-level map | do not carry | no gain in any model |
| V3 with posterior log-odds weights | do not carry | double-counts the prior; loses 13-53% of pairs |
| V3 with prior-free LLR weights on the published selection | **test next (predeclared, held out)** | post hoc: 82-97% of calibrated V0's pairs, 98-102% of its visible pairs, ESM-C F1 excess +0.021 [+0.006, +0.035] |
| Concave gap cost | do not carry | 3-11% effect, never decisive |
| V4 summed log-odds q-values | do not carry | accepts fewer or equal pairs than cos5 |
| Pair score | raw cosine (z2 for ESM2-8M) | within-list normalizations worse in every model |

The fair control for the recurrence-first architecture is **calibrated V0** (published pipeline +
built-in permutation null + per-size cosine cut), with **V1** as the filter-free baseline.

### Limitations
- The pair label is pairwise sequence visibility, so V2-V4 favour visible pairs partly by
  construction; chance ratio and F1 are the independent readouts. V0's 85,000 / 294,000 /
  148,000 accepted sequence-invisible pairs are not shown to be false here (structure check:
  jobs 76134 / 76209).
- The pooled isotonic map under-predicts seeds at the low end (esm2_8m lowest decile 0.12
  predicted vs 0.28 observed).
- 30 of 207 regions lie in held-out genomes; held-out F1 is not a separate test. Held-out
  visible fractions match all-genome ones within 0.005 for every accepted set.
- F1 excess intervals are about +/-0.035; smaller differences are unresolved.

### Artifacts
Local `benchmarks/redesign/ws_chain/`: `NOTEBOOK.md`; `code/` (`common.py`, `chain_v3.py`,
`extend_v2.py`, `prep.py`, `pairs.py`, `align_pairs.sh`, `fit_map.py`, `run_variants.py`,
`summarize.py`, `diag.py`, `render.py`, `unit_tests.py`, `run_*.sbatch`); `results/`
(`ABLATION.md`, `ablation_table.tsv`, `ablation_table_llr.tsv`, `diag.json`, `map_summary.json`,
per-model `summary.json`, `table.tsv`, `run_summary.json`, `map_report.json`, `prep_report.json`,
`map.json`; `align/` pair counts and recipe control).
biotite `$RD/ws_chain/`: `runs/<model>[_llr]/blocks__*.parquet`, `anchors__*.parquet` (paired
anchor rows with cosine and weight), `perm_blocks.parquet`, `selection__*.parquet`;
`results/<model>[_llr]/calls__*.parquet`, `per_region.parquet`; `work/<model>/` stats, pool,
map; `align/{pool,missing,missing_llr}/aln.tsv` (MMseqs2 labels, 7.25 M pairs).

### Open questions
- A predeclared, held-out test of V3 with LLR weights on the published selection (the one
  repair with a measured ESM-C F1 gain).
- A calibration that conditions on collinearity, P(visible | pair score, collinear support),
  in place of the pair-level map: the diagnostic places V0's extra yield there.
- Structural support for calibrated V0's sequence-invisible accepted pairs.

Correction (05:25 PST) to the closing report's carry table, row "Remove the binary support
gate": V1 and V2-p80 accept 29-39% and 27-35% fewer anchor pairs than calibrated V0
(V1 344,086 / 642,321 / 482,221; V2-p80 390,181 / 683,759 / 484,270), not 23-30%; the
23-27% figure applies to V2-p50.
