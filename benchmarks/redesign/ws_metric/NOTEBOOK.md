# ws_metric notebook

Workstream: pair-similarity function over frozen protein embeddings, including a metric
learned from synteny. Append-only. Timestamps are UTC.

Local dir: `benchmarks/redesign/ws_metric/` (this file, `code/`, small results, `REPORT.md`).
Remote dir: `RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111`,
working under `$RD/ws_metric/`.

---

## 2026-11-11T11:15Z — Setup and preflight

- Read `docs/redesign/BRIEF.md` in full.
- Slurm preflight: read `/home/jwestrob/.claude/CLAUDE.md` and
  `/home/jwestrob/jwestrob/obsidian-vault/biotite-cluster-ops.md` in full (same turn as
  the first sbatch will follow). Header to use: `-A banfield -p standard
  --qos=standard_expand -c 24`, exclude list, absolute `-o`, no `--mem`/`--time`; threads
  from `$SLURM_CPUS_ON_NODE`. Budget: 3 concurrent `standard_expand` jobs, at most one
  `standard_expand_2` job at a time.
- Starting script `$S/code/anchor_evidence.py` (job 76221) was PENDING at 11:08Z; its output
  `$S/results/anchor_evidence/anchor_evidence.json` does not exist yet. Jobs 76134 (full
  TM-align of 12.7 M pairs) and 76209 (key-subset TM-align, writes `$S/work_key/tm_key.parquet`)
  are RUNNING.
- MMseqs `align` on an explicit pair list is cheap: 12,676,988 pairs in 86 s on 64 threads
  (`$S/logs/seqinvis_76134.log`, 10:09:39Z to 10:11:05Z). Retrieval scoring for every arm by
  alignment is therefore affordable.

### Review of the starting script's joins (`anchor_evidence.py`)

Checked on biotite:
- `bgc_queries/metadata.parquet`: 67,148 rows, `row_index == position` for all rows, sample
  ids equal the universe `genome` for every query row. Reference metadata: 4,190,383 rows,
  `row_index == position`, genome equals the universe genome for every row. So
  `pd.factorize(sample_id)` codes indexed by `row_index` are aligned; the genome-mean join is
  correct.
- Held-out query genomes in the script: `q_names[::3]` of the alphabetically sorted names =
  **Kutzneria sp. 744 (10,200 proteins), Streptomyces albus J1074 (5,890), Streptomyces
  pristinaespiralis ATCC 25486 (6,869)** = 22,959 proteins. Training genomes: the other six.
- Seed sample: the hash rule and threshold match `prep_pairs.py` (`seed_unchained` uses the
  same `hash(qkey*5e6+tkey) % 1000 < thr_s`, and every chained pair was aligned), so the
  uniform seed sample should be fully scored. The script reports `popA_scored`; I assert it.
- Block anchor reversal for `orientation = -1` is applied before zipping (trap 2).
- Weak points, not join errors: (1) in-batch InfoNCE without masking, so a batch holding two
  positives of one orthologous family treats a true homolog as a negative; (2) the head is
  trained on every target phylum, so its numbers on distant targets are not held out;
  (3) `popB` is restricted to each model's own blocks, so models are compared on different
  pairs; (4) metrics on seeds only, with no fresh search.

## 2026-11-11T11:20Z — Plan and predeclared criteria (written before any result)

### Question
Does a projection trained on synteny anchors (positional-ortholog labels) give a better
pair-similarity function than the best untrained transform of the same frozen embeddings,
on held-out query genomes and on held-out target phyla?

### Data splits
- Held-out query genomes: the three listed above (every third, alphabetical). All
  evaluation uses only these queries. Restricted further to query proteins with a finite
  vector in all four models (`esm2_8m, esm2_650m, esmc600m, esmc6b`), so every arm and
  model sees identical queries.
- Held-out phyla: learned arms train only on pairs whose target genome is Actinomycetota
  (strata `0_genus_Streptomyces`, `1_same_class_other_genus`, `2_same_phylum_other_class`)
  and whose query genome is a training genome. Hard negatives come from those same
  genomes. Evaluation on "held-out phyla" = targets in stratum `3_other_phylum`
  (1,127 genomes, 3,562,289 proteins), which the learned arms never see.
- Unsupervised statistics (global mean, per-genome means, covariance/PCA, CSLS
  neighbourhood radii) are fitted on all reference proteins without labels.

### Arms (identical queries, identical reference, per model)
Untrained: `raw` cosine; `gcenter` (subtract global mean); `genomecenter` (subtract the
protein's own genome mean); `csls_q` (query-side local scaling, raw minus mean of the
query's top-10 reference cosines; pair score only, its within-query ranking equals raw);
`csls2` (two-sided CSLS, 2cos - r_q - r_t, r_t = mean top-10 cosine of the target to all
67,148 query proteins; retrieval is exact through augmented vectors [2q, -1]·[t, r_t]);
`abttK` (centre, remove top K principal components, K = 2, 10, 30); `zca10` (soft-ZCA,
alpha 0.1, from `elsa/transform.py`); `pca256` (centre, project on top 256 PCs; the
dimension-matched untrained control for the learned 256-d head). Pair-only extras on the
seed sample: `z_q`, `rank_q`.

Learned (input = globally centred vector, output L2-normalized; same map for both sides):
default `lin256_t05_ge6_inb` = linear to 256, temperature 0.05, positives = anchors of
blocks with >= 6 anchors, in-batch negatives with same-family masking. One factor varied
at a time: output 64 / 128 / 512; two-layer MLP (d -> 1024 -> 256, GELU); temperature
0.03 / 0.1; positives (ii) anchors of calibrated `accept05` blocks, (iii) the
sequence-visible subset of (i); negatives: in-batch plus 8 random proteins from the
positive's target genome and 8 from its query genome ("hard", lineage-matched).
Hyperparameters other than the ablated factor are fixed a priori (Adam 1e-3, batch 4,096,
fixed step budget) and never tuned on the held-out sets.

### Metrics (all on held-out query genomes)
1. Seed sample (E1): AUROC and AP for sequence visibility (MMseqs E <= 1e-3) on the uniform
   hash sample of the model's original ANN60 seeds; reported overall, in-phylum targets and
   other-phylum targets.
2. Cross-phylum in-block pairs (E2): AUROC/AP for sequence visibility on scaffold_k5
   calibrated in-block pairs whose target is in another phylum (fixed pair set across models).
3. TM (E3): AUROC for TM >= 0.5 among sequence-invisible in-block pairs (`tm_key.parquet`,
   union of all models' blocks, fixed pair set), split in-phylum / other-phylum.
4. Retrieval (E4, decisive): exact top-60 per held-out query over (a) the whole reference
   and (b) the other-phylum reference only, in each arm's space; MMseqs align of every
   retrieved pair with the `pipeline.sh` recipe. Report: fraction sequence-visible, median
   identity of visible pairs, queries with >= 1 visible hit, share per target stratum,
   share of all retrieved pairs that go to the single reference Streptomyces genome and are
   sequence-invisible (lineage leakage), and the fraction of native Spacedust pairs of these
   queries found in the top 60.

### Predeclared criteria
- **Gain**: a learned arm counts as a gain if, on held-out query genomes AND on held-out
  phyla, it raises the retrieval sequence-visible fraction by >= 5 percentage points OR the
  TM AUROC by >= 0.03 over the best untrained baseline of the same model. Concretely both
  must hold: (whole-reference retrieval vis fraction +5 pts or in-phylum TM AUROC +0.03) and
  (other-phylum-only retrieval vis fraction +5 pts or other-phylum TM AUROC +0.03).
- **Against**:
  (a) a gain confined to in-phylum targets (whole-reference retrieval or in-phylum AUROC up,
  other-phylum retrieval and other-phylum AUROC flat or down): the head learned the training
  lineage;
  (b) the untrained `pca256` (or any untrained arm) matching the learned arm within the
  margins: the gain is dimension reduction or centring, not synteny supervision;
  (c) seed-sample AUROC up while fresh-search retrieval is flat: re-ranking a fixed list
  improved, search did not;
  (d) a rise in visible fraction that comes with fewer queries having any visible hit or a
  lower Spacedust-pair recall: the metric concentrated on easy queries;
  (e) sequence-visible-only positives (iii) doing as well as synteny positives (i): synteny
  labels add nothing beyond sequence labels.
- **Lineage leakage** is judged on the share of retrieved pairs to the Streptomyces
  reference genome that are sequence-invisible; a metric that "strips lineage" must lower
  it relative to raw without lowering the visible fraction.

## 2026-11-11T11:45Z — Harness code and evaluation sets (job 76267)

Code (local `code/`, remote `$RD/ws_metric/code/`):
- `common.py`: paths, arm grid (`UNTRAINED`, `LEARNED`, `TRAIN`), `Transforms` (every arm as a
  torch transform; soft-ZCA formula copied from `elsa/transform.py`: cov shrunk
  `(1-a)C + a tr(C)/d I`, `W = V diag(lam^-1/2) V^T`, alpha 0.1).
- `prep_sets.py`: evaluation populations and training positives by universe gid.
- `run_model.py`: per model `stats` (one reference pass: finite mask, mean, per-genome means,
  covariance eigendecomposition, exact CSLS radii r_t and r_q), `train` (all learned arms),
  `eval` (exact top-60, whole reference and other-phylum reference, every arm in one pass;
  pair scores on the labelled populations).
- `summarize.py` + `run_align.sbatch`: union of retrieved pairs -> MMseqs `align -e inf
  --alignment-mode 3` against `keggclusterdb` (the `pipeline.sh` recipe) -> metrics.

Job 76267 (`run_prep.sbatch`, 24 CPU, 37 s) wrote `$RD/ws_metric/sets/`. Checks that passed:
universe query keys/rows/gene ids equal `$S/work/qx.parquet` for all 67,148 rows and reference
keys equal `$S/work/tx.parquet` for all 4,190,383 rows; every anchor pair derived from
`all_blocks.parquet` (with the orientation reversal) exists in that model's `anchors.parquet`
(0 missing, all four models); anchor-row counts are conserved through the gene-id join;
every seed-sample, positive and scaffold_k5 pair has an alignment (scored fraction 1.0).

| set | esm2_8m | esm2_650m | esmc600m | esmc6b |
|---|---|---|---|---|
| query vectors non-finite | 936 | 936 | 0 | 0 |
| popA seed sample, all queries | 460,217 | 472,374 | 463,832 | 474,231 |
| popA held-out queries | 156,905 | 161,023 | 158,446 | 161,793 |
| popA held-out, sequence-visible rate | 0.466 | 0.612 | 0.387 | 0.326 |
| popA held-out, other-phylum targets | 46,370 | 28,420 | 16,140 | 9,098 |
| blocks used for positives (>= 6 anchors or accept05) | 111,389 | 265,639 | 147,858 | 91,172 |
| train positives (i) >= 6 anchors | 150,441 | 220,310 | 179,075 | 140,917 |
| train positives (ii) accept05 | 346,908 | 662,852 | 446,604 | 312,780 |
| train positives (iii) visible subset of (i) | 133,090 | 179,472 | 151,134 | 123,442 |
| visible rate of (i) / (ii) | 0.885 / 0.847 | 0.815 / 0.724 | 0.844 / 0.787 | 0.876 / 0.817 |
| >= 6-anchor pairs dropped for other-phylum target | 840 | 2,564 | 1,153 | 678 |

Train positives = training query genomes x Actinomycetota target genomes. Every block with
>= 6 anchors is also `accept05` (the blocks count equals the accept05 count in BRIEF D).

Held-out query set for all evaluation: 22,959 held-out proteins, 22,621 with a finite vector
in all four models. Other populations (held-out queries only):
- `xphy` (scaffold_k5 calibrated in-block pairs, ESM2-8M retrieval): 560,992 pairs; by target
  stratum: Streptomyces 6,986 (vis 0.978), same class 239,455 (0.939), same phylum other class
  18,339 (0.877), **other phylum 291,452 (0.825)**, no GTDB 4,760 (0.766).
- `sd_heldout` native Spacedust pairs: 1,498,222 pairs over 16,174 queries; 1,082,336 to other-phylum targets.
- `popB` deferred: `$S/work_key/tm_key.parquet` not yet written (job 76209 running).

Fix: `sbatch --export` splits on commas, so the first smoke job 76268 received `STAGES=stats`
only. The wrapper now takes `+`-separated lists.

## 2026-11-11T11:25Z — Smoke run (job 76279, esm2_8m, 300,000 reference rows, 60 steps)

- Stats stage on the first 300,000 reference rows: 71 s. Explained variance of the top PCs
  (esm2_8m): 0.174, 0.117, 0.067, 0.045, ...; top 256 PCs hold 0.994 of the variance of 320.
  Mean vector norm 0.894 (raw ESM2-8M vectors share a large common component; consistent with
  BRIEF trap 4). Mean CSLS radius r_t = 0.956.
- r_q moved from the stats pass to the eval stage (mean of the raw arm's exact top-10); the
  column-wise top-k in the stats pass was the slow part and r_q is only needed for held-out
  queries.
- Training speed: 0.35 s per step at batch 4,096 for the 320-d linear head on 24 CPUs. Changed
  the fixed budget to batch 2,048 x 3,000 steps (6.1 M pair samples, 9 to 46 epochs of the
  positive sets) for wall time. Decided before any held-out number existed.

## 2026-11-11T11:35Z — Smoke completed; full stats+train jobs launched

- Smoke 76279 completed in 5 min 51 s (all stages, all 19 arms, popA 1,091 and xphy 4,175 pairs
  scored; popB absent as expected). Eval chunk 0 (262,144 rows, 2,262 queries, 19 arms): 39 s.
- Eval loop change: within each reference chunk the other-phylum rows are placed first, so the
  other-phylum top-60 is a top-k over a leading column slice; the earlier version wrote -inf into
  the non-other columns with an index_put. Smoke rerun 76306 checks that hits are identical to
  the first smoke (`hits_v1`), and 76307 tests `summarize.py` on the smoke hits (TAG=smoke).
- Full runs, stats+train only (eval to follow as separate dependent jobs, so eval code can
  still change): 76309 esm2_650m, 76310 esmc600m (standard_expand, 24 CPU), 76311 esmc6b
  (standard_expand_2, 64 CPU, the workstream's one allowed 64-CPU job), 76312 esm2_8m
  (queued after the smoke align 76307 to stay within 3 concurrent standard_expand jobs; a first
  submission 76308 was cancelled within seconds for that reason).

## 2026-11-11T11:40Z — Smoke checks passed; eval and align chained

- 76306 (smoke eval with reordered chunks) vs 76279 (`hits_v1`): for all 19 arms the hit sets
  agree with Jaccard >= 0.9999 and scores differ by <= 5.4e-7; the remaining differences are
  tie-breaks at rank 60 from float rounding. The reorder is a pure refactor.
- 76307 (smoke align + summarize): 650,222 unique retrieved pairs aligned in 7 s; every
  retrieved pair received an alignment (asserted); `ABLATION.md` written. Smoke numbers use a
  300,000-row reference and 60 training steps and are not results.
- Smoke observation to watch in the full run: the hard-negative arm (8 random same-genome
  negatives per side added to ~4,095 in-batch negatives) gave retrieval numbers identical to
  4 decimals to the in-batch default after 60 steps. Random same-genome proteins are rarely
  close after projection, so they contribute little to the softmax denominator.
- Pair scoring moved into `score_pairs` with a `pairs` stage (re-runnable per population,
  merges new arm columns), so popB can be scored once `tm_key.parquet` lands.
- Chained: eval 76318 (esm2_8m, after 76312), 76319 (esm2_650m, after 76309), 76320 (esmc600m,
  after 76310), 76321 (esmc6b, 64 CPU, after 76311); align+summarize 76322 after all four.
- Stats pass timing: 80-105 s per 262,144-row chunk (16 chunks) for every model.

## 2026-11-11T12:15Z — Stalled reads on node-64-768g-29; switch to block reads

- 76309 (esm2_650m stats) on node-64-768g-29 slowed from ~110 s to 786 s per chunk; the process
  sat in state D at 0% CPU, reading the mmap'd reference at 59 MB per 30 s (~2 MB/s,
  `/proc/<pid>/io`). `dd` of the same file from the login node with 64 MB blocks: 467-499 MB/s.
  Cancelled 76309 with its dependents 76319 and 76322.
- All vector access now goes through `common.RowReader` (np.fromfile at a byte offset, one large
  sequential read per 262,144-row block; scattered rows are read block-wise). Checked on the
  login node: identical arrays to the mmap path for a 2,000-row block and for scattered rows of
  both the esm2_650m and esmc600m reference and query matrices (NaN-aware equality).
- Resubmitted: 76365 esm2_650m stats+train (excluding node-64-768g-29 as well), 76366 its eval,
  76367 align+summarize after 76318, 76366, 76320, 76321.
- Stats so far (`$RD/ws_metric/full/stats/<m>/stats.json`): norm of the mean unit vector
  esm2_8m 0.894, esmc600m 0.855, esmc6b 0.826; first PC of the centred vectors explains
  0.171 / 0.389 / 0.209 of the variance; mean CSLS radius r_t 0.957 / 0.936 / 0.886.
- Training logs (3,000 steps, batch 2,048): esmc6b default head ends at train loss 0.072 vs
  validation 0.543 (validation = 5% of training-genome query proteins), esmc600m 0.416 vs 0.959,
  esm2_8m 0.782 vs 1.036. Wide train/validation gaps for the larger models.

## 2026-11-11T12:30Z — Exact search through faiss.knn

- The first full esm2_8m eval (76318) took 377 s per 262,144-row chunk with 19 arms (projected
  1.7 h); torch.topk over 2,048 x 262,144 score blocks dominated.
- `search_chunk` now calls `faiss.knn(..., METRIC_INNER_PRODUCT)` (exact brute force with heap
  selection) for the whole chunk and for the leading other-phylum slice. Login-node test
  (esm2_8m, 300 finite queries, 60,000 reference rows in three sub-chunks to exercise merging):
  identical top-60 sets and scores (max |diff| 0.0) to the torch path and to a single brute-force
  top-60, for both the whole set and the other-phylum subset; 0.20 s vs 0.33 s on 2 threads.
  (A first attempt included 5 query rows with NaN vectors; queries in the real eval are restricted
  to finite rows, `fin_all`.)
- Cancelled 76318 (12 min in) and 76367; resubmitted 76379 (esm2_8m eval) and 76380 (align after
  76379, 76366, 76320, 76321). The pending evals pick up the faiss path at start.

## 2026-11-11T12:45Z — Stats and training complete for esm2_8m, esmc600m, esmc6b (76312, 76310, 76311)

Unsupervised statistics (all reference proteins; `full/stats/<m>/stats.json`):

| model | d | finite reference rows | norm of mean unit vector | PC1 share of centred variance | variance in top 256 PCs | mean r_t |
|---|---|---|---|---|---|---|
| esm2_8m | 320 | 4,190,383 | 0.894 | 0.171 | 0.993 | 0.957 |
| esm2_650m (76365) | 1280 | 4,115,052 | 0.953 | 0.259 | 0.920 | 0.971 |
| esmc600m | 1152 | 4,190,383 | 0.855 | 0.389 | 0.975 | 0.936 |
| esmc6b | 2560 | 4,190,383 | 0.826 | 0.209 | 0.940 | 0.886 |

Training (3,000 steps x 2,048; `full/heads/<m>/train_summary.json`), final train / validation
InfoNCE loss (validation = 5% of training-genome query proteins, never the held-out genomes):

| arm | esm2_8m | esmc600m | esmc6b |
|---|---|---|---|
| lin256_t05_ge6_inb | 0.781 / 1.036 | 0.416 / 0.959 | 0.072 / 0.543 |
| lin64 | 0.825 / 1.108 | 0.503 / 1.042 | 0.136 / 0.624 |
| lin128 | 0.792 / 1.047 | 0.439 / 0.975 | 0.087 / 0.564 |
| lin512 | 0.777 / 1.033 | 0.407 / 0.949 | 0.068 / 0.537 |
| mlp256 | 0.783 / 1.045 | 0.462 / 0.999 | 0.120 / 0.585 |
| t0.03 | 0.805 / 1.087 | 0.350 / 1.040 | 0.020 / 0.628 |
| t0.10 | 1.324 / 1.482 | 0.907 / 1.239 | 0.620 / 0.840 |
| acc05 positives | 0.728 / 0.845 | 0.479 / 0.671 | 0.291 / 0.621 |
| visible positives | 0.101 / 0.190 | 0.005 / 0.020 | 0.002 / 0.020 |
| hard negatives | 0.784 / 1.038 | 0.417 / 0.964 | 0.073 / 0.544 |

Observations (training only, no held-out data):
- Visible-only positives are nearly trivial to separate (validation loss 0.02-0.19); the
  sequence-invisible 12-16% of the >= 6-anchor positives carry almost all of the loss.
- The `accept05` positive graph has one giant connected component (213,349 of 346,908 pairs for
  esm2_8m; 373,333 of 446,604 for esmc600m; 209,799 of 312,780 for esmc6b), against <= 5,188
  pairs for the >= 6-anchor graph. Small calibrated blocks link unrelated families. With
  same-family masking this removes most in-batch negatives for those pairs, which is why its loss
  is lower; the acc05 arm therefore differs from the default in both positives and effective
  negatives.
- Hard negatives (8 random same-genome proteins per side) leave the loss unchanged to 3 decimals:
  random same-genome proteins are not confusable after projection.

## 2026-11-11T13:00Z — Starting script results (job 76221, partial: esm2_8m and esm2_650m)

`$S/results/anchor_evidence/anchor_evidence.json` (esmc6b still running). Held-out query genomes
identical to mine. Its head: linear 256, t 0.05, 4 epochs (292 / 432 steps at batch 2,048),
positives = anchors of >= 6-anchor blocks of the six training genomes against every target
phylum, in-batch negatives without family masking. popB = invisible pairs in the model's own
blocks with a TM-score (job 76209 `tm_key.parquet`, 833,631 pairs). AUROC (AP):

| score | esm2_8m popA (seq vis) | esm2_8m popB (TM>=0.5) | esm2_650m popA | esm2_650m popB |
|---|---|---|---|---|
| raw | 0.680 (0.672) | 0.761 (0.461) | 0.780 (0.857) | 0.809 (0.473) |
| global centre | 0.774 (0.770) | 0.848 (0.621) | 0.757 (0.793) | 0.835 (0.393) |
| genome centre | 0.818 (0.812) | 0.880 (0.670) | 0.746 (0.793) | 0.842 (0.384) |
| csls_q | 0.539 (0.518) | 0.784 (0.492) | 0.613 (0.709) | 0.833 (0.475) |
| z_q | 0.569 (0.542) | 0.792 (0.505) | 0.599 (0.708) | 0.829 (0.498) |
| rank_q | 0.567 (0.533) | - | 0.593 (0.694) | - |
| synteny head | 0.867 (0.849) | 0.875 (0.667) | 0.896 (0.933) | 0.873 (0.563) |

Held-out popA n = 156,905 (esm2_8m), 161,023 (esm2_650m); popB held-out n = 47,665 / 95,392 with
TM>=0.5 rates (all genomes) 0.182 / 0.147. Precision of the top 20% of held-out seeds: raw 0.718
/ 0.935, synteny head 0.904 / 0.986. Joins verified as above; these numbers are consistent with
a head that re-ranks seed lists well. They contain no fresh search and no phylum hold-out.
`tm_key.parquet` sets (job 76209): accepted-block invisible pairs TM>=0.5 12.7% (569,464),
rejected-block invisible 7.6%, unchained invisible 5.6%, random 1.8%, accepted visible 97.1%.

## 2026-11-11T13:15Z — esm2_8m eval done (76379); popB set and preliminary summary (76397)

- 76379: 16 chunks in 2,633 s (~165 s per chunk, 19 arms, 22,621 queries); popA 156,905 and
  xphy 560,992 pairs scored for every arm.
- `tm_key.parquet` landed (76209 COMPLETED, 833,631 pairs). 76397 (one slot, sequential):
  `prep_sets.py --only popB`, popB pair scores for esm2_8m, then `run_align.sbatch` (align +
  summarize over whatever hits exist, i.e. esm2_8m) for an early look.
- `run_align.sbatch` now scores popB for every evaluated model lacking it before collecting
  pairs (one slot, sequential). Final align 76380 depends on 76379, 76366, 76320, 76321, 76397.

## 2026-11-11T13:25Z — RESULT, esm2_8m round 1 (76379 eval, 76397 align + summarize)

popB set (`prep_sets.py --only popB`): 247,904 sequence-invisible in-block pairs of held-out
queries with a TM-score (union of the 7 models' blocks); TM>=0.5 rate 0.115; by target stratum
Streptomyces 10,053 (0.080), same class 212,944 (0.111), same phylum other class 6,227 (0.121),
other phylum 17,639 (0.185). esm2_8m scores 246,862 of them (finite targets).
76397 aligned 7,148,005 unique retrieved pairs? -> see `full/align/collect.json`; every retrieved
pair received an alignment (asserted).

Retrieval, exact top-60 for 22,621 held-out queries (vis = MMseqs E <= 1e-3; `all` = whole
reference, `other` = other-phylum reference only). Local copy:
`results/prelim_esm2_8m/{ABLATION.md,retrieval.tsv,pair_metrics.tsv,*_ci.tsv}`.

| arm | vis all | vis other | queries with a visible other-phylum hit | strep invisible share | SD recall | TM AUROC in / other | popA AUROC in / other | xphy AUROC other |
|---|---|---|---|---|---|---|---|---|
| raw | 0.466 | 0.330 | 0.660 | 0.0070 | 0.088 | 0.825 / 0.768 | 0.689 / 0.641 | 0.761 |
| gcenter | 0.476 | 0.334 | 0.661 | 0.0070 | 0.089 | 0.868 / 0.814 | 0.759 / 0.776 | 0.796 |
| genomecenter | 0.485 | 0.370 | 0.679 | 0.0055 | 0.090 | 0.895 / 0.840 | 0.810 / 0.801 | 0.710 |
| csls2 | 0.463 | 0.334 | 0.665 | 0.0050 | 0.086 | 0.839 / 0.790 | 0.672 / 0.629 | 0.772 |
| abtt2 | 0.489 | 0.372 | 0.677 | 0.0064 | 0.092 | 0.905 / 0.842 | 0.811 / 0.823 | 0.808 |
| abtt10 | 0.496 | 0.380 | 0.679 | 0.0065 | 0.094 | 0.903 / 0.869 | 0.817 / 0.851 | 0.838 |
| abtt30 | 0.538 | 0.420 | 0.701 | 0.0047 | 0.105 | 0.906 / 0.877 | 0.843 / 0.848 | 0.798 |
| **zca10** | **0.559** | **0.433** | 0.709 | 0.0054 | 0.111 | **0.907 / 0.878** | 0.870 / **0.873** | 0.833 |
| pca256 | 0.473 | 0.332 | 0.659 | 0.0070 | 0.089 | 0.868 / 0.814 | 0.756 / 0.774 | 0.794 |
| lin256_t05_ge6_inb (default) | 0.568 | 0.424 | 0.717 | 0.0043 | 0.114 | 0.869 / 0.838 | 0.884 / 0.848 | 0.807 |
| lin64 | 0.548 | 0.405 | 0.706 | 0.0044 | 0.111 | 0.867 / 0.832 | 0.874 / 0.836 | 0.802 |
| lin128 | 0.564 | 0.420 | 0.715 | 0.0043 | 0.114 | 0.868 / 0.837 | 0.882 / 0.845 | 0.806 |
| lin512 | 0.569 | 0.426 | 0.718 | 0.0043 | 0.115 | 0.869 / 0.838 | 0.885 / 0.848 | 0.809 |
| mlp256 | 0.560 | 0.413 | 0.711 | 0.0045 | 0.113 | 0.869 / 0.835 | 0.883 / 0.846 | 0.804 |
| t0.03 | 0.557 | 0.395 | 0.704 | 0.0050 | 0.113 | 0.846 / 0.797 | 0.880 / 0.829 | 0.721 |
| t0.10 | 0.576 | 0.444 | 0.720 | 0.0037 | 0.116 | 0.900 / 0.862 | 0.880 / 0.850 | 0.810 |
| acc05 positives | 0.557 | 0.407 | 0.709 | 0.0048 | 0.113 | 0.864 / 0.821 | 0.873 / 0.827 | 0.792 |
| visible positives | 0.577 | 0.442 | 0.722 | 0.0035 | 0.117 | 0.884 / 0.856 | 0.867 / 0.830 | 0.817 |
| hard negatives | 0.568 | 0.424 | 0.717 | 0.0043 | 0.115 | 0.869 / 0.838 | 0.884 / 0.848 | 0.807 |

Best untrained arm on every decisive metric: soft-ZCA alpha 0.1 (`zca10`).
Retrieval differences vs zca10 (query bootstrap 95% CI): best learned, whole reference
`lin256_t05_vis_inb` +1.83 pts [+1.67, +1.96], `lin256_t10_ge6_inb` +1.76 [+1.62, +1.89], default
+0.91 [+0.78, +1.04]; other-phylum search `lin256_t10_ge6_inb` +1.16 [+1.02, +1.28], visible
positives +0.91 [+0.76, +1.05], default -0.81 [-0.94, -0.69]. zca10 vs raw: +9.20 pts all,
+10.25 pts other. Pair AUROC best learned minus best untrained (query-cluster bootstrap):
popA in-phylum +0.015 [+0.012, +0.018]; popA other-phylum -0.024 [-0.026, -0.020]; xphy
in-phylum +0.028 [+0.016, +0.042]; xphy other-phylum -0.021 [-0.039, -0.002]; TM in-phylum
-0.007 [-0.009, -0.004]; TM other-phylum -0.016 [-0.022, -0.011].

Against the predeclared criteria (esm2_8m): **no learned arm is a gain.** The largest retrieval
gains over zca10 are +1.8 pts (whole reference) and +1.2 pts (other phylum), below the 5-point
margin; TM AUROC is lower than zca10 in both strata. The pattern matches the predeclared
"against" case (a): the learned heads win only on in-phylum pair AUROCs (popA, xphy) and lose on
every other-phylum pair AUROC. Untrained whitening supplies most of the improvement over raw
cosine (+9.2 / +10.3 pts), and pca256 (dimension reduction alone) adds +0.6 / +0.2 pts.
Lineage leakage is small for this model (0.70% of raw retrieved pairs go to the Streptomyces
reference genome and are invisible; 0.35-0.54% for whitening and learned arms).
Other observations: higher temperature (0.10) and visible-only positives are the two best
single factors; t 0.03 is worst; hard negatives change nothing (identical to the default to 4
decimals); output dimension 128-512 is flat, 64 loses ~2 pts.

Round 2 (exploratory, chosen after seeing these esm2_8m numbers, so any win among them is
subject to selection; one seed replicate added to size training variance), job 76398:
`lin256_t10_vis_inb` (both best factors), `zlin256_t10_ge6_inb` and `zlin256_t10_vis_inb`
(head on the soft-ZCA input), `lin256_t20_ge6_inb` (temperature axis extended),
`lin256_t10_ge6_inb_s18` (seed 18 replicate of t0.10). Final align 76380 now waits for 76398.

Correction to the entry above (placeholder line): 76397 aligned **8,930,055** unique retrieved
(query, target) pairs of esm2_8m (19 arms x 2 searches x 22,621 queries x 60) in 108 s on 24
threads; 8,930,055 alignments returned.

## 2026-11-11T13:45Z — Round 2 queued for all models; zca10 vectors exported early

- 76398 (esm2_8m round 2): 5 heads trained in 14 min, eval of 5 arms 11 min; pair files merged
  (all round-1 columns kept). Round 2 queued on the other models after their round-1 eval:
  76401 esmc600m, 76402 esmc6b (64 CPU), 76403 esm2_650m; this doubles as an out-of-sample check
  of arms chosen on esm2_8m. Final align+summarize 76380 waits for all of them, then the TM sample
  of invisible retrieved pairs 76399.
- Fix before it bit: a round-2 eval writes `pairs_popB.parquet` with round-2 columns only for
  models whose round-1 eval preceded `popB.parquet`; `run_align.sbatch` now recomputes popB when
  the file lacks the `raw` column (all arms, merged).
- `report_tables.py`: per-model tables plus the predeclared verdict (best learned minus best
  untrained on whole-reference vis, other-phylum vis, TM AUROC in and other phylum; round 1 alone
  and rounds 1+2), appended to the align and TM jobs.
- 76442 (`run_apply.sbatch`): soft-ZCA (alpha 0.1) vectors for esm2_650m and esmc600m in gid order
  to `$RD/ws_metric/vectors/<model>__zca10.npy` + `.json`, exported now so the tokens workstream can
  start; zca10 is the strongest untrained transform on every decisive esm2_8m metric. The best
  learned arm will be exported too if the full results warrant it.

## 2026-11-11T13:50Z — zca10 vectors written and checked (76442)

- `$RD/ws_metric/vectors/esm2_650m__zca10.npy` (4,257,531 x 1,280 float32, 21.8 GB; 76,267 NaN
  rows = 75,331 reference + 936 query proteins without an ESM2 vector) and
  `esmc600m__zca10.npy` (4,257,531 x 1,152, 19.6 GB; 0 NaN rows), each with a `.json` manifest
  (inputs, stats file, row order, code hash 79579ca6e0368916). 140 s and 191 s.
- Check: for gids 0, 123456, 2,000,000, 4,190,382 (last reference), 4,190,383 (first query) and
  4,257,530 (last query), the stored row equals the transform recomputed from the source vectors
  to <= 2.6e-7; rows are unit norm.

## 2026-11-11T14:25Z — RESULT, esmc6b round 1 (eval 76321; interim align+summary 76457)

76457 aligned 25,124,754 unique retrieved pairs (esm2_8m rounds 1+2, esmc6b round 1) in 3.5 min.
Local copy: `results/interim_8m_6b/` (REPORT_TABLES.md, retrieval.tsv, pair_metrics.tsv, CIs).
popB for the three larger models was scored inside their round-1 evals (pairs are scored after
the reference pass, and `popB.parquet` existed by then).

| arm | vis all | vis other | qany other | strep invisible share | SD recall | TM AUROC in / other | popA in / other | xphy other |
|---|---|---|---|---|---|---|---|---|
| raw | 0.320 | 0.137 | 0.544 | 0.0706 | 0.082 | 0.629 / 0.570 | 0.763 / 0.454 | 0.778 |
| gcenter | 0.317 | 0.132 | 0.531 | 0.0841 | 0.082 | 0.686 / 0.588 | 0.720 / 0.376 | 0.777 |
| genomecenter | 0.374 | 0.216 | 0.614 | 0.0289 | 0.092 | 0.755 / 0.629 | 0.728 / 0.385 | 0.786 |
| csls2 | 0.342 | 0.172 | 0.585 | 0.0501 | 0.088 | 0.700 / 0.640 | 0.755 / 0.572 | 0.774 |
| abtt30 | 0.454 | 0.332 | 0.684 | 0.1123 | 0.109 | 0.804 / 0.805 | 0.864 / 0.743 | 0.865 |
| **zca10** | 0.609 | **0.519** | **0.763** | 0.0897 | 0.138 | 0.857 / **0.849** | 0.925 / **0.835** | **0.930** |
| pca256 | 0.273 | 0.106 | 0.493 | 0.0814 | 0.071 | 0.683 / 0.583 | 0.677 / 0.340 | 0.766 |
| lin256 default | 0.577 | 0.304 | 0.679 | 0.0123 | 0.140 | 0.823 / 0.740 | 0.894 / 0.647 | 0.856 |
| lin256 t0.10 | 0.667 | 0.471 | 0.756 | 0.0086 | 0.149 | 0.848 / 0.756 | 0.920 / 0.621 | 0.879 |
| lin256 visible positives | **0.668** | 0.448 | 0.757 | **0.0078** | 0.148 | **0.871** / 0.807 | **0.938** / 0.767 | 0.864 |
| lin64 | 0.476 | 0.199 | 0.599 | 0.0101 | 0.130 | 0.808 / 0.720 | 0.885 / 0.619 | 0.844 |
| mlp256 | 0.512 | 0.228 | 0.635 | 0.0135 | 0.133 | 0.813 / 0.716 | 0.889 / 0.612 | 0.832 |

(abtt2/10, lin128/512, t0.03, acc05 and hard negatives in `REPORT_TABLES.md`; hard negatives again
equal the default to 3 decimals.)

Differences (query bootstrap / query-cluster bootstrap, 95% CI):
- zca10 vs raw: +28.9 pts whole reference [+28.5, +29.3]; +38.3 pts other phylum [+37.8, +38.7].
- Best learned vs zca10, whole reference: visible positives +5.85 pts [+5.58, +6.13]; t0.10 +5.77
  [+5.47, +6.06]; default -3.25 [-3.51, -2.98].
- Best learned vs zca10, other-phylum search: t0.10 -4.80 [-5.03, -4.57]; visible positives -7.14
  [-7.40, -6.90]; default -21.5 [-21.9, -21.1].
- TM AUROC (visible-positive head minus zca10): in-phylum +0.014 [+0.011, +0.017]; other phylum
  -0.041 [-0.051, -0.033]. popA: in +0.013 [+0.011, +0.015], other -0.069 [-0.079, -0.059].
  xphy other (t0.10 head): -0.051 [-0.063, -0.037].

Verdict against the predeclared criteria (esmc6b): the held-out-genome criterion is met (+5.85
pts whole-reference visible fraction), the held-out-phyla criterion fails in every metric
(-4.8 to -7.1 pts retrieval, -0.041 TM AUROC). **Not a gain; this is predeclared "against" case
(a): the gain is confined to the training lineage.** The learned heads do strip lineage leakage:
the share of retrieved pairs that are sequence-invisible hits to the single Streptomyces genome
falls from 7.06% (raw) and 8.97% (zca10) to 0.78% (visible-positive head), and genome centring
alone brings it to 2.89%. Soft-ZCA raises leakage slightly while raising visibility everywhere.

## 2026-11-11T15:30Z — RESULT, round 1 complete for all four models; round 2 for three (interim 76473)

76473 aligned 42,529,029 unique retrieved pairs in 5.4 min (all round-1 arms of four models,
round 2 of esm2_8m, esmc600m, esmc6b). Local copy: `results/interim2/`. esm2_650m round 2 (76403)
is still in eval; the final align+summary 76380 follows it.

Retrieval visible fraction, whole reference / other-phylum reference (held-out queries):

| arm | esm2_8m | esm2_650m | esmc600m | esmc6b |
|---|---|---|---|---|
| raw | 0.466 / 0.330 | 0.612 / 0.409 | 0.388 / 0.143 | 0.320 / 0.137 |
| genomecenter | 0.485 / 0.370 | 0.642 / 0.488 | 0.432 / 0.244 | 0.374 / 0.216 |
| csls2 | 0.463 / 0.334 | 0.614 / 0.433 | 0.401 / 0.170 | 0.342 / 0.172 |
| abtt30 | 0.538 / 0.420 | 0.703 / 0.561 | 0.560 / 0.390 | 0.454 / 0.332 |
| pca256 | 0.473 / 0.332 | 0.573 / 0.350 | 0.357 / 0.121 | 0.273 / 0.106 |
| **zca10** | 0.558 / 0.433 | **0.726 / 0.588** | 0.651 / 0.514 | 0.609 / 0.519 |
| r1 default head | 0.568 / 0.424 | 0.703 / 0.518 | 0.649 / 0.410 | 0.577 / 0.304 |
| r1 best head (t0.10 or visible positives) | 0.577 / 0.444 | 0.720 / 0.566 | 0.700 / 0.525 | 0.668 / 0.471 |
| r2 lin256_t10_vis_inb | 0.579 / 0.448 | pending | 0.730 / 0.570 | 0.716 / 0.546 |
| r2 zlin256_t10_ge6_inb (zca input, synteny positives) | 0.578 / 0.446 | pending | 0.705 / 0.533 | 0.671 / 0.478 |
| **r2 zlin256_t10_vis_inb** (zca input, visible positives) | 0.580 / 0.449 | pending | **0.735 / 0.578** | **0.726 / 0.563** |

TM AUROC on sequence-invisible in-block pairs (in-phylum / other-phylum targets):

| arm | esm2_8m | esm2_650m | esmc600m | esmc6b |
|---|---|---|---|---|
| raw | 0.826 / 0.768 | 0.783 / 0.709 | 0.640 / 0.572 | 0.629 / 0.570 |
| abtt30 | 0.906 / 0.877 | **0.919** / 0.913 | 0.815 / 0.805 | 0.805 / 0.804 |
| zca10 | 0.907 / 0.878 | 0.912 / **0.914** | 0.859 / 0.845 | 0.858 / 0.848 |
| r1 best head | 0.901 / 0.862 | 0.875 / 0.864 | 0.869 / 0.802 | 0.871 / 0.807 |
| r2 zlin256_t10_vis_inb | 0.905 / 0.868 | pending | 0.896 / 0.842 | 0.903 / 0.862 |

Query-bootstrap 95% CIs, zlin256_t10_vis_inb minus zca10, visible fraction: esm2_8m +2.17 pts
[+2.02, +2.30] all, +1.62 [+1.48, +1.75] other; esmc600m +8.43 [+8.17, +8.71] all, +6.46 [+6.22,
+6.68] other; esmc6b +11.71 [+11.39, +12.01] all, +4.38 [+4.16, +4.61] other.

Verdicts against the predeclared criteria (`REPORT_TABLES.md`):
- Round 1 (the predeclared grid): **no gain for any model.** esm2_650m: the best head trails zca10
  by 0.6 / 2.2 pts and by 0.044 / 0.050 TM AUROC. esmc600m: +4.9 / +1.1 pts, TM -0.043 other.
  esmc6b: +5.85 / -4.8 pts, TM other -0.041 (in-lineage only, case (a)). esm2_8m: +1.8 / +1.2.
- Round 2 (post hoc, chosen on esm2_8m): `zlin256_t10_vis_inb` meets both criteria on **esmc600m**
  (+8.4 / +6.5 pts; TM +0.037 in, -0.003 other) — an out-of-sample model for that choice; on
  esmc6b it meets the held-out-genome criterion (+11.7 pts, TM +0.046) and misses the phylum
  criterion narrowly (+4.4 pts, TM +0.014); on esm2_8m it is +2.2 / +1.6.
- Predeclared "against" case (e) holds everywhere: heads trained on the sequence-visible subset of
  synteny anchors beat heads trained on all synteny anchors (whose extra ~12-16% are invisible
  positional pairs) on retrieval and on TM AUROC. The useful supervision is sequence homology;
  synteny supplies the candidate pairs.
- Case (b) is excluded: pca256 (untrained 256-d) is worse than raw for esm2_650m, esmc600m and
  esmc6b, so the head's gain is not dimension reduction.
- Lineage leakage (share of retrieved pairs that are invisible hits to the single Streptomyces
  genome): raw 0.70% / 1.16% / 3.79% / 7.06% (8m / 650m / 600m / 6b); zca10 0.54% / 0.53% /
  5.01% / 8.97%; heads 0.32-0.78%. Every learned head strips most of it; whitening does not.
- Seed replicate (t0.10, seed 18 vs 17): visible fraction within 0.001 (esm2_8m 0.576 vs 0.576,
  esmc600m 0.700 vs 0.700, esmc6b 0.668 vs 0.667), so training variance is far below the effects.

Exported (76488, running): `zlin256_t10_vis_inb` vectors for esmc600m and esm2_650m (256-d) beside
the zca10 vectors.

Correction to the entry above: the learned-head leakage range is 0.32-1.88%, not 0.32-0.78%. The
`accept05`-positive heads leak most (esmc600m 1.27%, esmc6b 1.88%); every other head is 0.32-0.90%.

Second correction (exact, from `retrieval.tsv`, whole-reference search): learned-head leakage
ranges are esm2_8m 0.34-0.50%, esm2_650m 0.32-0.70%, esmc600m 0.35-1.27%, esmc6b 0.52-1.88%; the
minimum is `zlin256_t10_vis_inb` (or the visible-positive head) and the maximum the `accept05` head
in every model. Untrained transforms: esmc6b raw 7.06%, zca10 8.97%, abtt30 11.23%, genomecenter
2.89%, csls2 5.01%; esmc600m raw 3.79%, zca10 5.01%, genomecenter 2.48%. The previous correction's
"0.32-0.90%" was also wrong.

## 2026-11-11T15:35Z — Learned-head vectors written and checked (76488)

- `$RD/ws_metric/vectors/esmc600m__zlin256_t10_vis_inb.npy` and `esm2_650m__zlin256_t10_vis_inb.npy`
  (4,257,531 x 256 float32, 4.36 GB each; NaN rows 0 / 76,267), manifests beside them (arm config:
  linear 256 on the soft-ZCA input, t 0.10, sequence-visible >= 6-anchor positives of the six
  training genomes x Actinomycetota targets, in-batch negatives). Rows at six gids (first/last
  reference, first/last query, two interior) equal the recomputed transform to <= 2.5e-7.
- Manifest copies: `results/vectors/*.json`. Local `code/common.r1.py` and `code/run_model.r1.py`
  are snapshots taken before the round-2 options were added; the deployed code equals `code/*.py`
  (md5 match, local vs remote, all 13 files).

## 2026-11-11T16:00Z — RESULT, final align + summary (76380): all arms, all four models

- 76380 aligned 43,019,147 unique retrieved pairs (5 min, 24 threads), every retrieved pair
  scored; summary 22 min. Local: `results/final/` (`REPORT_TABLES.md`, `ABLATION.md`,
  `retrieval.tsv`, `pair_metrics.tsv`, `retrieval_diff_ci.tsv`, `pair_auroc_diff_ci.tsv`,
  `verdict.tsv`, `cross_model_tables.md`).
- Note: Slurm stores the batch script at submission, so 76380 and 76399 ran the versions of
  `run_align.sbatch`/`run_tm.sbatch` from 04:26 PST (no popB loop, no `report_tables.py` call).
  popB was already complete for all four models; I ran `report_tables.py` by hand on the login
  node afterwards (seconds).
- Labelled populations (held-out queries; identical pairs across models where the population is
  model-independent): popA n = 156,905 (esm2_8m; per model ~156-161 k), vis rate 0.466; xphy
  559,160 pairs (in-phylum 263,863 vis 0.937; other phylum 290,559 vis 0.826); popB 245,132
  invisible in-block pairs (in-phylum 226,588 TM>=0.5 rate 0.111; other phylum 17,508 rate 0.185).

esm2_650m round 2 (the last missing piece): `zlin256_t10_vis_inb` vs zca10 +0.54 pts [+0.43,
+0.65] whole reference, -0.29 pts [-0.40, -0.17] other phylum; TM AUROC -0.011 in (vs abtt30) and
-0.016 other. No gain for esm2_650m in either round.

Cross-model summary (full tables in `results/final/cross_model_tables.md`), visible fraction whole
/ other-phylum reference:

| arm | esm2_8m | esm2_650m | esmc600m | esmc6b |
|---|---|---|---|---|
| raw | 0.466 / 0.330 | 0.612 / 0.409 | 0.388 / 0.143 | 0.320 / 0.137 |
| zca10 | 0.558 / 0.433 | 0.726 / 0.588 | 0.651 / 0.514 | 0.609 / 0.519 |
| best round-1 head | 0.577 / 0.444 | 0.720 / 0.566 | 0.700 / 0.525 | 0.668 / 0.471 |
| zlin256_t10_vis_inb (round 2) | 0.580 / 0.449 | 0.732 / 0.585 | 0.735 / 0.578 | 0.726 / 0.563 |

Final verdicts (predeclared criteria; `verdict.tsv`):

| model | round 1 (predeclared grid) | rounds 1+2 |
|---|---|---|
| esm2_8m | no (+1.8 / +1.2 pts; TM -0.007 / -0.016) | no (+2.2 / +1.6 pts) |
| esm2_650m | no (-0.6 / -2.2 pts; TM -0.044 / -0.050) | no (+0.5 / -0.3 pts; TM -0.011 / -0.016) |
| esmc600m | no (+4.9 / +1.1 pts; TM +0.010 / -0.043) | **yes** (+8.4 / +6.5 pts; TM +0.037 / -0.003) |
| esmc6b | no, in-lineage only (+5.9 / -4.8 pts; TM +0.014 / -0.041) | genomes yes, phyla no (+11.7 / +4.4 pts; TM +0.046 / +0.014) |

Interpretation, limited to what the numbers show:
1. Untrained soft-ZCA whitening (alpha 0.1) is the largest single improvement over raw cosine in
   every model: +9.2 / +10.3 pts (esm2_8m), +11.4 / +17.9 (esm2_650m), +26.3 / +37.1 (esmc600m),
   +28.9 / +38.3 (esmc6b) visible fraction, whole / other-phylum reference, with TM AUROC gains of
   +0.08 to +0.28. The more anisotropic ESM-C embeddings gain most. Removing the top 30 PCs
   (abtt30) recovers most but not all of it; global centring, two-sided CSLS and genome centring
   recover little; pca256 is at or below raw.
2. The synteny-trained metric of the predeclared grid does not beat whitening on held-out phyla
   for any model. Where it wins (esmc6b whole reference, in-phylum pair AUROCs) it loses on
   other-phylum targets: predeclared case (a), learned training lineage.
3. The one configuration that beats whitening on both held-out sets (esmc600m) and on held-out
   genomes for esmc6b is a head stacked on the whitened input and trained on the sequence-visible
   subset of synteny anchors at temperature 0.10. It was chosen after seeing esm2_8m round-1
   results (post hoc), was confirmed out of sample on esmc600m, missed the phylum margin on
   esmc6b by 0.6 pts, and gave nothing on ESM2 models. Its supervision is sequence homology
   restricted to positional pairs; the invisible positional pairs that synteny adds lower every
   metric (case (e)).
4. Every learned head strips the lineage leakage that whitening leaves in ESM-C (esmc6b share of
   retrieved pairs that are invisible hits to the single Streptomyces genome: raw 7.1%, zca10
   9.0%, round-2 head 0.5%).
5. Seed replicate: visible fraction identical to within 0.001; hard negatives (random
   same-genome proteins): no effect in any model; output dimension 128-512 flat, 64 worse; MLP no
   better than linear; temperature 0.10 > 0.05 > 0.03, 0.20 below 0.10.

Correction to point 5 above: temperature 0.20 vs 0.10 (synteny positives, gcenter input) is
-1.3 / -0.9 pts (esm2_8m), -0.4 / +0.1 (esm2_650m), -0.1 / +0.7 (esmc600m) and +0.6 / +1.7
(esmc6b): flat overall, not uniformly below 0.10. Point 1 detail: genome centring adds +1.9 / +4.0
(esm2_8m), +3.0 / +7.9 (esm2_650m), +4.4 / +10.1 (esmc600m), +5.4 / +7.9 pts (esmc6b) over raw.

## 2026-11-11T16:25Z — RESULT, structure of sequence-invisible retrieved pairs (76399)

Hash sample of ~1,500 sequence-invisible retrieved pairs with AFDB structures per (model, arm,
search), 192 groups, 77,842 unique pairs, Foldseek TM-align (the `pipeline.sh` recipe; 27 min on
24 threads), TM = max of the two normalizations. `results/final/retrieval_tm.tsv`; tables in
`results/final/cross_model_tables.md`. Binomial 95% half-width at n = 1,500 is about 0.025.
TM >= 0.5 means shared fold, kept apart from orthology (BRIEF trap 9).

Fraction TM >= 0.5 among invisible retrieved pairs, whole / other-phylum reference:

| arm | esm2_8m | esm2_650m | esmc600m | esmc6b |
|---|---|---|---|---|
| raw | 0.350 / 0.351 | 0.384 / 0.334 | 0.143 / 0.094 | 0.141 / 0.110 |
| zca10 | 0.378 / 0.415 | 0.491 / 0.560 | 0.277 / 0.342 | 0.262 / 0.346 |
| abtt30 | 0.389 / 0.430 | 0.480 / 0.564 | 0.219 / 0.209 | 0.190 / 0.212 |
| lin256 default | 0.387 / 0.414 | 0.458 / 0.406 | 0.301 / 0.225 | 0.217 / 0.165 |
| zlin256_t10_vis_inb | 0.379 / 0.414 | 0.516 / 0.574 | 0.482 / 0.490 | 0.486 / 0.521 |

Estimated share of retrieved pairs with sequence or fold support (visible + invisible x TM>=0.5
fraction; assumes the structured invisible pairs represent all invisible pairs):

| arm | esm2_8m | esm2_650m | esmc600m | esmc6b |
|---|---|---|---|---|
| raw | 0.653 / 0.565 | 0.761 / 0.606 | 0.476 / 0.223 | 0.416 / 0.232 |
| zca10 | 0.725 / 0.668 | 0.861 / 0.819 | 0.748 / 0.680 | 0.712 / 0.686 |
| zlin256_t10_vis_inb | 0.739 / 0.677 | 0.870 / 0.823 | 0.863 / 0.785 | 0.859 / 0.791 |

Reading: for ESM-C, the round-2 head's extra retrievals are not sequence-visible pairs alone; its
invisible hits share a fold 1.5-1.8x as often as zca10's (esmc600m 0.48 vs 0.28 whole, 0.49 vs 0.34
other phylum; esmc6b 0.49 vs 0.26, 0.52 vs 0.35). For ESM2 the two are within ~0.03. Raw ESM-C
invisible hits share a fold at 0.09-0.14, close to the 0.018 random-pair rate scaled by the
in-block baseline (0.127 for accepted-block invisible pairs, job 76209), which matches the BRIEF
finding that raw ESM-C cosine retrieves many non-homologous pairs.

Correction to the last sentence above (it was garbled): raw ESM-C invisible hits share a fold at
0.09-0.14, comparable to the 0.127 rate among invisible pairs of accepted blocks and above the
0.018 rate of random pairs (both from job 76209).

## 2026-11-11T16:35Z — REPORT (closing entry; `REPORT.md` could not be written from this agent)

The agent harness refused a `REPORT.md` write from this subagent ("subagents should return
findings as text"). This entry is the report; the coordinator can copy it to `REPORT.md`.

### What was tested
Pair-similarity functions over frozen mean-pooled embeddings of esm2_8m, esm2_650m, esmc600m and
esmc6b, on 22,621 held-out query proteins (Kutzneria sp. 744, S. albus J1074, S.
pristinaespiralis; finite in all four models) against the 4,190,383-protein reference.
- Untrained: raw, global centring, per-genome centring, query-side and two-sided CSLS, top-PC
  removal (2/10/30), soft-ZCA alpha 0.1, PCA-256; z-score and rank as pair-only scores.
- Learned (predeclared grid, round 1): linear 64/128/256/512, MLP, temperature 0.03/0.05/0.10,
  positives from >= 6-anchor blocks / accept05 blocks / the sequence-visible subset, in-batch vs
  added same-genome negatives; trained only on six training genomes x Actinomycetota targets.
- Round 2 (post hoc after esm2_8m round 1; flagged): both best factors combined, heads on the
  soft-ZCA input, temperature 0.20, one seed replicate.
- Metrics: fresh exact top-60 retrieval (whole reference and other-phylum-only reference) scored by
  MMseqs2 visibility (43,019,147 aligned pairs), lineage leakage, Spacedust-pair recall,
  TM-align of invisible retrieved pairs; AUROC/AP on the seed sample, scaffold_k5 cross-phylum
  in-block pairs, and invisible in-block pairs labelled TM >= 0.5.

### Ablation table
`results/final/REPORT_TABLES.md` (every arm x model, with the predeclared verdict),
`results/final/cross_model_tables.md` (key arms side by side), CIs in
`results/final/retrieval_diff_ci.tsv` and `pair_auroc_diff_ci.tsv`.

### What works
1. Soft-ZCA whitening (alpha 0.1): best untrained transform on every decisive metric in every
   model. Over raw cosine: +9.2 / +10.3 (esm2_8m), +11.4 / +17.9 (esm2_650m), +26.3 / +37.1
   (esmc600m), +28.9 / +38.3 pts (esmc6b) retrieval visible fraction, whole / other-phylum
   reference; TM AUROC +0.08 to +0.28; Spacedust-pair recall +0.02 to +0.07.
2. Head on the whitened input trained on the sequence-visible subset of >= 6-anchor synteny pairs,
   t 0.10 (`zlin256_t10_vis_inb`, round 2): beyond zca10, esmc600m +8.4 / +6.5 pts (meets both
   predeclared criteria), esmc6b +11.7 / +4.4 pts (held-out genomes only), esm2_650m +0.5 / -0.3,
   esm2_8m +2.2 / +1.6. Its invisible ESM-C hits share a fold 1.5-1.8x as often as zca10's.
3. Every learned head removes lineage leakage: esmc6b invisible hits to the single Streptomyces
   genome 7.1% (raw), 9.0% (zca10), 0.5% (round-2 head).

### What does not work
- The predeclared learned grid (round 1) is not a gain for any model; on esmc6b its gain is
  confined to the training lineage (+5.9 pts whole reference, -4.8 pts other phylum, TM AUROC
  -0.041 other phylum).
- Synteny-only supervision: adding the sequence-invisible positional pairs to the positives lowers
  retrieval and TM AUROC in every model (predeclared case (e)). Calibrated accept05 blocks as
  positives are worse still (their positive graph has a giant component of 210-373 k pairs).
- Random same-genome hard negatives: no measurable effect. MLP: no better than linear.
  Output 64-d: worse; 128-512: flat.
- Two-sided CSLS (vs query set), global centring, PCA-256: at most +3 pts; PCA-256 below raw for
  three models. Query-side scaling, z-score and rank lower AUROC on seed lists.

### Artifacts
- Vectors (gid order, float32, `.json` manifests): `$RD/ws_metric/vectors/esm2_650m__zca10.npy`,
  `esmc600m__zca10.npy` (full width), `esm2_650m__zlin256_t10_vis_inb.npy`,
  `esmc600m__zlin256_t10_vis_inb.npy` (256-d). For esmc600m the 256-d head is the best transform;
  for esm2_650m zca10 and the head are equivalent (+0.5 / -0.3 pts).
- Per-arm hits, pair scores, heads, stats: `$RD/ws_metric/full/{eval,heads,stats}/<model>/`.
- Code: `code/` (local and `$RD/ws_metric/code/`, md5-identical).

### Open questions
- Whether the round-2 head's esmc600m gain survives a held-out choice of configuration (it was
  chosen on esm2_8m results over the same held-out genomes) and other held-out genome sets.
- Whether the other-phylum shortfall on esmc6b (+4.4 pts) closes with positives from more
  phyla; positives here were Actinomycetota-only by design.
- Downstream effect on tokens and blocks (tokens workstream can test the four vector files).

## 2026-11-11T16:40Z — Addenda

- Starting script 76221 completed (16 min 44 s). esmc6b held-out AUROC (AP): popA raw 0.745
  (0.590), global centre 0.698 (0.499), genome centre 0.708 (0.505), csls_q 0.546, z_q 0.636,
  rank_q 0.634, synteny head 0.879 (0.742); popB (TM>=0.5, own blocks) raw 0.619 (0.187), global
  centre 0.688, genome centre 0.758 (0.242), csls_q 0.726, z_q 0.750, synteny head 0.808 (0.288).
  Its pattern (head >> centring on seed lists) agrees with my popA in-phylum numbers; my harness
  adds what it lacked: soft-ZCA (0.925 popA in-phylum for esmc6b, above its head), the phylum
  hold-out (where its kind of head falls below whitening) and fresh-search retrieval.
- Correction to the closing REPORT entry: two-sided CSLS reaches +3.5 pts over raw (esmc6b other
  phylum), not "at most +3"; global centring stays within +1.0 pt; PCA-256 is below raw for
  esm2_650m, esmc600m and esmc6b.
- Not done, by the BRIEF's "write nowhere else" rule: no vault capture was written outside the
  workstream directories.
