# ELSA redesign: coordinator notebook

Append-only. Times are UTC unless marked. Biotite paths use
`ROOT=~/srvp2020/Jacob/ELSA/spacedust`, `S=$ROOT/seq_invisible_20261111`,
`RD=~/srvp2020/Jacob/ELSA/redesign_20261111`. Shared briefing for all workstreams:
[BRIEF.md](BRIEF.md).

---

## 2026-11-11 09:40 — Starting point

Jacob asked for bearings on the ELSA paper: how to improve it, how to test it, and
how to find compelling examples. State found:

- Three manuscript drafts exist. March (`ELSA_Manuscript.pdf`,
  `benchmarks/evaluation/manuscript/ELSA_manuscript_v3.md`): claims voided by the July
  audit. September (`docs/manuscript/ELSA_manuscript.md`): accurate, headline is a
  71-comparison E. coli table, no biology section.
- Data on local disk: no DPANN run (deleted 14 Sept), Omnitrophota sidecar from 18 July
  (before the clustering fix), SR-VP sidecars from 23 Oct.
- The September headline on the Spacedust collection was F1 0.757 (ESM2-8M) to 0.819
  (ESM-C 6B) against 0.676 for native Spacedust, with random placements of the
  annotations scoring 0.78-0.79 against ELSA's blocks.

Jacob authorized free exploration on biotite.

## 10:00 — Experiment 1: sequence-invisible, structure-confirmed pairs

Question: what share of ELSA's chained anchor pairs lies outside pairwise sequence
detection, and what share of those is confirmed by structure? Criteria written before
results in `benchmarks/evaluation/spacedust_2026-09-09/seq_invisible_20261111/README.md`
(P1 invisible share >= 5%; P2 structure-confirmed share of invisible pairs >= 50%;
P3 chained vs unchained ratio >= 1.5; the claim is abandoned if P1 < 2% everywhere or
P2 < 25%).

Jobs: 76130 and 76132 failed in the built-in smoke test within seconds (traps 1 and 2
of the brief: row index vs key; reverse-orientation list order). 76134 passed smoke and
ran the full set: 12,676,988 unique pairs from seven models, native Spacedust and
random controls. MMseqs2 Smith-Waterman on all pairs took 86 s on 64 CPUs. TM-align
on 11,253,665 pairs with structures runs at about 0.1 s per pair (5-6 h).

Result, sequence step (`results/EARLY_SEQ_SUMMARY.md`):

| Model | Chained pairs | Sequence-invisible | Pairs also in Spacedust |
|---|---|---|---|
| ESM2 8M | 1,482,427 | 47.0% | 26.4% |
| ESM2 35M | 1,421,881 | 46.3% | 30.4% |
| ESM2 150M | 1,423,774 | 47.5% | 31.5% |
| ESM2 650M | 1,513,349 | 46.3% | 32.4% |
| ESM-C 300M | 1,545,998 | 53.0% | 27.4% |
| ESM-C 600M | 1,377,536 | 51.6% | 29.9% |
| ESM-C 6B | 1,435,130 | 57.5% | 24.4% |
| Native Spacedust | 4,359,060 | 9.3% | 100% |
| Random pairs | 300,000 | 99.9% | 0% |

By block size (ESM2-8M): invisible 63.3% (2 anchors), 50.6% (3), 33.2% (4), 17.6% (5-9),
7.5% (10+). By target taxonomy: 87-97% of chained pairs target Actinomycetota genomes,
which hold 13.9% of reference proteins; Spacedust places 72% of its pairs in other phyla.
ESM-C 6B sends 8.85% of pairs to the one Streptomyces genome, 77% of them invisible.

P1 is met many times over, which made the question "are they real".

## 10:15 — Experiment 2: permutation null (and a correction)

`block_fdr.py` (job 76157) re-chained saved anchors after permuting query positions and
reported 0.5-1.4% chance blocks. **This null is invalid**: the anchors had been selected
by the neighbourhood step using the true gene order. Recorded in the experiment README.
The valid null permutes before selection. The September panels do that; from them
(ESM2-8M, paired-overlap arm):

| Block size | Observed | Permuted mean | Ratio |
|---|---|---|---|
| 2 | 346,039 | 399,219 | 1.154 |
| 3 | 106,073 | 86,396 | 0.814 |
| 4 | 39,400 | 9,251 | 0.235 |
| 5 | 18,179 | 423 | 0.023 |
| 6+ | 25,299 | 5.7 | 0.0002 |

`block_fdr.py`'s rescoring of the 207 regions by block-size threshold is valid and
matches later runs.

## 10:20 — Experiment 3: calibration by mean anchor cosine

`score_calibration.py` (job 76171), then `permuted_panels.py` (array 76175) for all seven
models. The latter replays each observed run from saved seeds (block and anchor counts
match exactly for all seven) and regenerates three whole-pipeline permutations; totals
for ESM2-8M (495,161 / 495,380 / 495,342) and ESM-C 6B (535,642 / 532,999 / 534,641)
equal the September panels.

Rule: for each block size, the smallest cut on mean anchor cosine such that
(permuted blocks above the cut) / (observed above the cut) <= 5%.

| Model | Calibrated blocks | Pair visibility | F1 | F1 random | Excess [95% CI] |
|---|---|---|---|---|---|
| ESM2 8M | 111,389 | 0.842 | 0.6519 | 0.6580 | -0.006 [-0.041, +0.027] |
| ESM2 35M | 174,383 | 0.794 | 0.6930 | 0.7125 | -0.020 [-0.049, +0.009] |
| ESM2 150M | 237,830 | 0.733 | 0.7487 | 0.7452 | +0.004 [-0.019, +0.025] |
| ESM2 650M | 265,639 | 0.721 | 0.7544 | 0.7594 | -0.005 [-0.028, +0.017] |
| ESM-C 300M | 140,066 | 0.814 | 0.7443 | 0.7049 | +0.040 [+0.014, +0.065] |
| ESM-C 600M | 147,858 | 0.784 | 0.7624 | 0.7147 | +0.048 [+0.023, +0.072] |
| ESM-C 6B | 91,172 | 0.813 | 0.7490 | 0.6685 | +0.081 [+0.054, +0.107] |
| Native Spacedust | 1,641,181 | 0.907 | 0.6765 | 0.6115 | +0.065 [+0.037, +0.093] |

Paired differences on calibrated blocks: 35M - 8M +0.041 [+0.017, +0.067];
150M - 35M +0.056 [+0.037, +0.077]; 650M - 150M +0.006 [-0.010, +0.022];
650M - 8M +0.103 [+0.076, +0.130]; ESM-C 6B - ESM2 650M -0.005 [-0.033, +0.022].

Accepted vs rejected small blocks (ESM2-8M): size 2 accepted 12,114 at cut 0.9881,
97.1% pair-visible, rejected 34.5%; size 3 accepted 25,748, 86.6% vs 37.5%; size 4
accepted 30,049, 76.8% vs 34.6%.

## 10:40 — Experiment 4: retrieval budget

`retrieval_compare.py` (job 76218), ESM2-8M, three retrieval rules, each with its own
permutation panels, each calibrated to 5%:

| Retrieval | Blocks | Permuted/observed | Calibrated | Cross-phylum calibrated | F1 calibrated |
|---|---|---|---|---|---|
| global top-60, no selection | 116,931 | 0.012 | 116,931 | 4,572 | 0.510 |
| neighbourhood selection (production) | 534,990 | 0.926 | 111,389 | 4,474 | 0.652 |
| top-5 per reference scaffold | 2,147,019 | 0.576 | 373,150 | 186,529 | 0.660 |

Per-scaffold retrieval reaches 1,125 other-phylum genomes (410 for production), gives
12,233 cross-phylum blocks of 10+ anchors (7 for production), and covers 35,873 of
67,148 query genes with a calibrated cross-phylum block (6,412 for production).
Plain global top-60 has almost no chance blocks: the chance blocks come from the
neighbourhood selection.

## 11:00 — Code reading

Read `seed.py`, `neighborhood.py`, `chain.py`, `cluster.py`, `embeddings.py`, `params.py`
in `ELSA-production` and the research checkout.

- `neighborhood.py` ranks supported candidates above all unsupported ones
  (`np.lexsort((union, -similarity, -context, -supported))`), and a candidate is
  supported when its cosine is >= 0.85. That gate is satisfied by nearly every pair.
- `chain.py` maximizes chain length first, summed cosine second (`new_len > best_len or
  (new_len == best_len and new_score > best_score)`).
- `embeddings.py` mean-pools the last layer; long proteins are averaged over windows.
- The two checkouts have diverged: production lacks commit `d65a16e` (the clustering
  fix; it still has `jaccard_tau` 0.3 and `mutual_k` 5), research lacks `e7810bb`
  (neighbourhood retrieval and paired-overlap extraction).

## 11:10 — Decision (Jacob)

Jacob authorized all proposed changes, asked for proper ablation and thorough records,
and stated a preference for a new architecture over the MCScanX-plus-chaining lineage.
Later messages: use many Opus subagents; prioritize biotite; the H200 is down, so CPU
only.

## 11:15 — Program

Target: **recurrence-first discovery**. Family tokens per protein; token pairs counted
within a few genes across all genomes; significance from recurrence against a
within-contig shuffle; loci read from the per-genome support landscape.

Workstreams and owners are in BRIEF section 3. Six Opus agents launched at about 11:20:
tokens, metric, chain, seqanchors, dpann, priorart. The coordinator owns the engine.

Shared gene universe built by `benchmarks/redesign/build_universe.py` (job 76225):
4,257,531 genes, 1,317 genomes, 2,061 contigs, at `$RD/universe/`.

## 11:30 — Engine v0

`benchmarks/redesign/engine/recurrence.py`. Statistics per unordered token pair:
G genomes, L genera, NP phyla, J = G / genomes carrying the rarer token, E expected
genomes under random order (sum over genomes of min(1, 2w * copies_a * copies_b / genes)),
P = Poisson tail. Null: tokens shuffled within each contig, three replicates. Rules on
(L, J, P) are scored by null-passing / observed-passing; among rules at FDR <= 1% the one
covering the most genes is chosen. Loci: maximal runs of boundaries with support >= m
for m in 2..800 ("conserved"), and unsupported runs flanked by supported boundaries
("islands"). Scoring: `evaluate` on the 207 regions, 200 random placements.

Predeclared reading of the first runs:
- The design is viable if, at FDR <= 1%, significant token pairs cover a majority of
  genes, and conserved loci reach an F1 near the calibrated pairwise range (0.65-0.76)
  with a non-negative excess over random placement.
- It has a clear advantage if cross-phylum support covers far more query genes than the
  6,412 reached by production retrieval, at no retrieval cost.
- It fails if tokens are too coarse or too fragmented to give recurrence: few
  significant pairs, or F1 well under 0.5 at every token resolution.

First run: development tokens `dev__esm2_8m__raw__K16384` (single-level k-means, coarse;
job 76248) then engine job 76249 at w = 3. The tokens workstream delivers the real arms.

Still running from the earlier experiments: 76134 (full TM-align), 76209 (TM-align on
569,464 sequence-invisible pairs in calibrated blocks plus controls), 76221 (pair-score
comparison, queued behind 76209), 76144 and 76201 (taxonomy summary and final join,
queued behind 76134).

## 2026-11-12 00:10 — Gap in this notebook

Between 11:30 and 00:10 UTC the coordinator session did not write entries. The record
for that period is the agents' notebooks, the job outputs, and this consolidated entry,
written from them. Jacob returned at about 00:05 UTC.

## 00:15 — Structure result for sequence-invisible pairs (jobs 76209, 76134)

Job 76134 finished in 10 h 57 min (TM-align 5 h 39 min, exact TM recomputation 4 h 57 min).
Job 76209 (decisive subset) finished in 2 h 8 min.

Calibration of the readout: sequence-visible pairs in calibrated blocks are 97.1%
structure-confirmed (TM >= 0.5 by the shorter chain); random pairs 1.8%.

| Set | Pairs with structures | Structure-confirmed |
|---|---|---|
| Sequence-invisible, in calibrated blocks (7 models) | 569,464 | 12.7% |
| Sequence-invisible, in rejected blocks (sample) | 151,536 | 7.6% |
| Sequence-invisible, unchained retained anchors (sample) | 40,020 | 5.6% |
| Native Spacedust pairs, sequence-invisible | 403,491 | 93.4% |

All chained pairs, P2 by model: ESM2 8M 13.7%, 35M 11.2%, 150M 10.2%, 650M 10.0%;
ESM-C 300M 7.5%, 600M 6.8%, 6B 7.9%. In blocks of 10+ anchors P2 is 12.1-14.7%.
Original seeds outside blocks: 33-41% for ESM2 models, 13-15% for ESM-C.

**Verdict on the predeclared criteria (experiment 1):** P1 met (46-58% invisible), P2
failed (7-14%, bar 25%). The claim "embedding anchors add remote homology that sequence
anchors miss" is abandoned for this collection and this pipeline. Sequence-invisible
anchors inside real blocks are mostly non-homologous occupants of matching positions.
ELSA chains 0.7-1.0% of Spacedust's sequence-invisible pairs and 11-12% of its visible
ones.

The final join (job 76201) failed on a SQL alias; fixed and rerun as 77562.

## 00:20 — Pair-score comparison (job 76221)

Held-out query genomes (3 of 9). AUROC for sequence visibility on a uniform seed sample,
and for TM >= 0.5 among sequence-invisible in-block pairs:

| Score | ESM2-8M vis / TM | ESM2-650M vis / TM |
|---|---|---|
| raw cosine | 0.680 / 0.761 | 0.780 / 0.809 |
| global centring | 0.774 / 0.848 | 0.757 / 0.835 |
| per-genome centring | 0.818 / 0.880 | 0.746 / 0.842 |
| query-side local scaling | 0.539 / 0.784 | 0.613 / 0.833 |
| linear head trained on anchors of blocks with 6+ anchors | 0.867 / 0.875 | 0.896 / 0.873 |

The metric workstream replaced this with a proper harness (below).

## 00:25 — Workstream results (agents)

Each workstream's full record is its `NOTEBOOK.md`; `REPORT.md` holds the closing report
(saved by the coordinator for tokens, metric, chain and priorart, whose own writes the
harness blocked; seqanchors wrote its own; dpann's report is its notebook and package
README).

**priorart.** The statistical core of the recurrence design is published: Rogozin 2002
(31 genomes, gene-order permutation null), Junier & Rivoire 2016 (1,445 genomes,
lineage-weighted, binomial test, "synteny segments"), CSBFinder-S 2020 (1,485 genomes,
near-linear). Hotspots between conserved flanks and co-occurrence modules are published
(Oliveira 2017, panRGP, Hochhauser 2023, panModule). k-means family tokens from ESM-2
are published (Bacformer, 50,000 centroids). No prior found for embedding-derived tokens
feeding a recurrence test at 10^3 genomes, or for a similarity metric supervised by
synteny. "A new architecture" as a headline would be overreach.

**tokens (56 arms).** MMseqs2 30%/80% clusters: 1,038,303 tokens, pair homology precision
0.9997, recall 0.555 on sequence-visible Spacedust pairs (0.450 cross-phylum). k-means on
embeddings fails as a family stand-in in all 30 arms (recall <= 0.159, homology precision
<= 0.77). Graph communities on ESM2-650M reach homology precision 0.81-0.86 at recall
0.20-0.24. Hybrid (sequence families merged by ESM2-650M centroid similarity, Leiden
CPM): recall 0.799 (0.698 cross-phylum) at pair homology precision 0.966, with 60% of
tokens pure. Recommendation: sequence clusters as the primary tokens, the CPM hybrid
(`...mm30cpmr0p05`) as the arm for distant recurrence.

**metric.** Soft-ZCA whitening (alpha 0.1) is the best transform for every model. Fresh
exact top-60 search, fraction of hits that are sequence-visible, whole reference /
other-phylum targets: ESM2-8M 0.466 -> 0.558 / 0.330 -> 0.433; ESM2-650M 0.612 -> 0.726 /
0.409 -> 0.588; ESM-C 600M 0.388 -> 0.651 / 0.143 -> 0.514; ESM-C 6B 0.320 -> 0.609 /
0.137 -> 0.519. The synteny-learned head from the predeclared grid does not beat
whitening on held-out phyla for any model; where it wins it wins inside the training
lineage. Adding sequence-invisible synteny pairs to the positives lowers every metric.
One later configuration (linear head on whitened input, sequence-visible positives)
beats whitening on ESM-C 600M (+8.4 / +6.5 points). Whitened and learned vectors for all
genes are at `$RD/ws_metric/vectors/`.

**chain.** The binary support gate causes the chance blocks: raw permuted/observed is
0.926 / 0.666 / 0.795 with it (ESM2-8M / ESM2-650M / ESM-C 600M) and 0.011 / 0.009 / 0.019
without it. No predeclared repair (evidence-gated selection, score-first chaining with
log-odds weights, score-based q-values) beats the published pipeline plus the per-size
cosine cut. Carry into the package: the built-in three-permutation null with that cut.

**seqanchors (24 arms).** Sequence anchors through the same chainer have a clean null
with no calibration: permuted/observed 0.005 (top 60 per query) and 0.033 (every hit).
Every-hit sequence anchors give the best F1 excess of all arms, +0.073 [+0.052, +0.094],
and 2.49 M cross-phylum blocks (16,975 of 10+ anchors) against 17,544 (51) for the best
embedding arm. Every embedding-only arm is below it in paired F1 excess by 0.025-0.094.
Embedding seeds added to sequence anchors keep the null clean and contribute an
estimated 31,800 (ESM2-650M) / 13,800 (ESM-C 600M) structure-confirmed pairs with no
sequence hit: 0.3-0.7% of accepted pairs, no F1 gain. All of this used raw cosine.

**dpann.** Package at `$RD/ws_dpann/package/`: 3,585,303 genes, 3,564 genomes, gene order
trusted for all, 57,310 curated gold instances (DefenseFinder 4,495; TXSScan 809;
CRISPRCasTyper 52,006), 674,172 sequence-family tokens, a scorer passing 32 self-tests.
**The March DPANN embeddings are defective.** The writer (Sharur `fc03c29`,
`src/ingest/06_esm2_embeddings.py`) averaged over every position of a padded batch of 16,
so each vector includes its batch-mates' padding. Re-running the exact batches reproduces
the stored vectors (median cosine 1.0000, 94% >= 0.999); a clean single-protein embedding
sits at median cosine 0.93 from the stored vector, the level of unrelated proteins. The
code was fixed on 30 March (`4592775`). The Omnitrophota store is dated 10 March.

## 00:30 — Actions

- Engine on real token arms: jobs 77563-77572 (sequence clusters at w = 1, 2, 3, 5 and
  oriented at w = 3; two hybrid arms; ESM2-650M graph; two k-means arms as negative
  controls).
- dpann agent resumed: test the Omnitrophota store and every other Sharur embedding store
  for the same defect. No re-embedding started.
- seqanchors agent resumed: whitened-embedding anchor arms (exact top-60 on the zca10
  vectors, alone and merged with sequence anchors), same tables, same structure check,
  criterion predeclared in its notebook.

## 00:55 — Engine v0 on real token arms (jobs 77563-77572)

Window w = 3 unless stated; three within-contig shuffles; rule chosen as the one covering
the most genes among rules at FDR <= 1%. "Conserved loci" = level sets of the support
landscape, no gap bridging. Each run takes 3-7 minutes on 24 CPUs, most of it benchmark
scoring.

| Tokens | Significant pairs | FDR | Genes covered | Query genes with other-phylum support | F1 conserved loci | Excess [95% CI] | F1 all locus kinds | Excess |
|---|---|---|---|---|---|---|---|---|
| sequence clusters 30%/80%, w 1 | 138,580 | 0.05% | 42.4% | 10,235 | 0.426 | +0.009 [-0.027, +0.046] | 0.610 | +0.019 |
| sequence clusters, w 2 | 215,047 | 0.94% | 46.2% | 12,862 | 0.495 | +0.028 [-0.011, +0.068] | 0.656 | +0.028 |
| sequence clusters, w 3 | 273,903 | 0.27% | 47.2% | 13,936 | 0.532 | +0.044 [+0.004, +0.085] | 0.672 | +0.024 |
| sequence clusters, w 3, strand-aware | 294,889 | 0.93% | 47.2% | 13,326 | 0.526 | +0.041 [+0.000, +0.081] | 0.667 | +0.018 |
| sequence clusters, w 5 | 377,875 | 0.33% | 48.5% | 15,180 | 0.592 | +0.093 [+0.053, +0.133] | 0.700 | +0.037 |
| hybrid, CPM 1e-4 | 255,977 | 0.31% | 48.3% | 15,180 | 0.540 | +0.047 [+0.007, +0.087] | 0.670 | +0.021 |
| hybrid, CPM r 0.05 | 206,063 | 0.49% | 54.4% | 22,384 | 0.623 | +0.100 [+0.065, +0.135] | 0.699 | +0.050 |
| ESM2-650M graph communities, r 1000 | 273,480 | 0.74% | 53.2% | 10,019 | 0.607 | +0.084 [+0.047, +0.119] | 0.699 | +0.046 |
| k-means ESM2-650M genome-centred, K 262,144 | 298,013 | 0.06% | 32.7% | 1,299 | 0.385 | -0.030 [-0.071, +0.010] | 0.605 | -0.015 |
| k-means ESM2-8M raw, K 262,144 | 143,241 | 0.10% | 16.5% | 610 | 0.223 | -0.057 [-0.095, -0.019] | 0.484 | -0.043 |

Reading against the criteria predeclared at 11:30:
- Viable: yes for sequence, hybrid and graph tokens (significant pairs cover 47-54% of
  genes at FDR under 1%; conserved loci score 0.53-0.62 with a positive excess). The F1
  itself is under the calibrated pairwise range (0.65-0.76) and under every-hit sequence
  chaining (0.727).
- Advantage on distant support: 13,936 to 22,384 query genes carry a significant
  adjacency that also occurs in another phylum, against 6,412 for production retrieval
  and 35,873 for exhaustive per-scaffold retrieval.
- The excess over random placement of hybrid-token conserved loci (+0.100) is the largest
  measured for any method here (Spacedust +0.065; every-hit sequence chaining +0.073;
  ESM-C 6B calibrated +0.081).
- k-means tokens fail, as the tokens workstream found. Graph communities built on the
  same embeddings work: token construction, not the embedding, was the problem.
- Islands alone do not localize BGCs (excess about 0 or negative).

Engine v1 (`recurrence.py` rewritten): universe directory as an argument, modules
(components of tokens joined by significant pairs with J >= 0.5), gap-bridging loci, an
all-genome locus table. Jobs 77591-77595 rerun the leading arms with v1 into
`engine/results_v1/`; job 77590 runs v1 on the DPANN package with sequence-family tokens
and then the gold-locus scorer. Two more permissive sequence-family token sets
(E <= 1e-3, coverage 80% or 50%, no identity floor) are clustering (77584, 77585) with
engine runs behind them (77586, 77587).

## 01:25 — Engine v1 results, and two changes before the next runs

v1 on the Spacedust collection (jobs 77591-77595; within-contig shuffle null):

| Tokens, window | Conserved loci, gaps 0-2: F1 | Random | Excess [95% CI] | All locus kinds: F1 | Excess |
|---|---|---|---|---|---|
| sequence clusters, w 3 | 0.571 | 0.525 | +0.046 [+0.004, +0.085] | 0.698 | +0.022 |
| sequence clusters, w 5 | 0.622 | 0.528 | +0.094 [+0.053, +0.133] | 0.725 | +0.039 |
| hybrid CPM 1e-4, w 3 | 0.580 | 0.530 | +0.049 [+0.009, +0.089] | 0.698 | +0.021 |
| hybrid CPM r 0.05, w 3 | 0.672 | 0.572 | +0.099 [+0.064, +0.133] | 0.742 | +0.053 |
| hybrid CPM r 0.05, w 5 | 0.737 | 0.576 | +0.161 [+0.127, +0.193] | 0.788 | +0.089 |

Bridging gaps adds 0.04-0.05 F1 at unchanged excess. Modules as defined (components of
tokens joined by significant pairs with J >= 0.5) have a giant component of 21,000-57,000
tokens beside thousands of small ones; the module definition needs work.

v1 on DPANN (job 77590; sequence-family tokens, w 3, within-contig shuffle): the rule
that met FDR 1% was L >= 10 and J >= 0.4, only 8,209 significant pairs covering 25.7% of
genes. Gold loci with at least two members, best-single-locus F1 against 200 random
placements: TXSScan 0.657 vs 0.348 (804 instances; 84% covered at 50%); DefenseFinder
0.401 vs 0.216 (2,472); CRISPRCasTyper 0.268 vs 0.229 (3,820).

**Change 1, the null.** DPANN has 231,477 contigs for 3,585,303 genes. Shuffling within
a contig of a few genes leaves most gene pairs within the window, so the null barely
differs from the data and the FDR target forces a very strict rule. v1.1 shuffles within
each genome (contig sizes kept), the null of Rogozin 2002. On the Spacedust collection
(2,061 contigs, mostly complete genomes) the two nulls should agree; one arm is run with
both to measure that.

**Change 2, the window is tuned on the benchmark it is scored on.** The F1 gain from
w 3 to w 5 was seen on source207, so it is a tuned value there. Predeclared check on
independent data: the DPANN gold loci. If the summed F1 excess for DefenseFinder and
TXSScan at w 5 is at least that at w 3, w 5 becomes the default; otherwise the window
stays a per-dataset setting and both are reported. Runs: Spacedust hybrid r 0.05 at
w 3, 5, 8, 12; sequence clusters at w 5, 8; ESM2-650M graph at w 5; DPANN at w 3, 5, 8.

## 2026-11-12 00:57 UTC — Window sweep with the genome-level null (jobs 77649-77659, 77586-77587)

(Clock note: the headings from "00:10" to "01:25" above were stamped ahead of the real UTC
time; this heading and later ones use `date -u`.)

All 13 jobs completed. Engine v1.1, shuffle within genome, rule chosen at FDR <= 1%.

Spacedust collection, source207, conserved loci with gaps 0-2 (F1, random, excess [95% CI])
and all locus kinds:

| Tokens | w | Rule | Significant pairs | Conserved g0-2: F1 | Random | Excess | All kinds: F1 | Excess |
|---|---|---|---|---|---|---|---|---|
| hybrid CPM r 0.05 | 3 | P<=1e-4 & L>=3 | 206,063 | 0.672 | 0.572 | +0.099 [+0.064, +0.133] | 0.742 | +0.053 |
| hybrid CPM r 0.05 | 5 | same | 270,979 | 0.737 | 0.576 | +0.161 [+0.127, +0.193] | 0.788 | +0.089 |
| hybrid CPM r 0.05 | 8 | same | 329,844 | 0.750 | 0.566 | +0.184 [+0.154, +0.213] | 0.801 | +0.099 |
| hybrid CPM r 0.05 | 12 | same | 373,688 | 0.714 | 0.544 | +0.170 [+0.137, +0.202] | 0.796 | +0.099 |
| sequence clusters 30%/80% | 5 | same | 377,875 | 0.622 | 0.528 | +0.094 [+0.053, +0.133] | 0.725 | +0.039 |
| sequence clusters 30%/80% | 8 | same | 484,760 | 0.645 | 0.516 | +0.129 [+0.090, +0.167] | 0.750 | +0.061 |
| ESM2-650M graph r 1000 | 5 | same | 363,702 | 0.693 | 0.573 | +0.120 [+0.086, +0.153] | 0.779 | +0.078 |
| sequence families E<=1e-3, cov 50% | 3 | same | 228,812 | 0.692 | 0.571 | +0.121 [+0.085, +0.156] | 0.754 | +0.064 |
| sequence families E<=1e-3, cov 80% | 3 | same | 260,583 | 0.617 | 0.533 | +0.084 [+0.043, +0.123] | 0.723 | +0.044 |

- **The two nulls agree on this collection.** Hybrid r 0.05 at w 5 with the within-contig
  shuffle (job 77653): same rule, the same 270,979 pairs, identical F1 (0.7366 / 0.5760);
  empirical FDR 0.0056 against 0.0052.
- **w 8 is the best window here** (excess +0.184); w 12 loses precision.
- **A permissive sequence family is as good as the hybrid tokens at equal window.**
  E <= 1e-3 with 50% coverage (601,584 tokens) at w 3 gives F1 0.692, excess +0.121,
  against hybrid r 0.05 at w 3 with 0.672, +0.099. The hybrid arm's gain over the 30%/80%
  clusters is therefore largely a family-breadth effect that MMseqs alone delivers. Next
  comparison: the E <= 1e-3 / 50% arm at w 5 and w 8 against the hybrid arm at the same
  windows.

DPANN package (sequence clusters 30%/80%, 674,172 tokens; rule P <= 1e-4 & L >= 2):

| w | Significant pairs | Genes covered | DefenseFinder F1 / control / excess | TXSScan | CRISPRCasTyper |
|---|---|---|---|---|---|
| 3 | 176,590 | 55.4% | 0.720 / 0.463 / +0.258 | 0.731 / 0.548 / +0.183 | 0.527 / 0.466 / +0.061 |
| 5 | 222,206 | 56.8% | 0.698 / 0.466 / +0.231 | 0.727 / 0.555 / +0.172 | 0.530 / 0.470 / +0.060 |
| 8 | 276,099 | 57.9% | 0.687 / 0.472 / +0.215 | 0.716 / 0.554 / +0.162 | 0.527 / 0.474 / +0.053 |

(instances with at least two members: DefenseFinder 2,472, TXSScan 804, CRISPRCasTyper 3,820)

- **The genome-level null repaired the DPANN run.** Against the within-contig shuffle
  (8,209 pairs, 25.7% of genes, DefenseFinder F1 0.401): 176,590 pairs, 55.4% of genes,
  DefenseFinder F1 0.720.
- **Predeclared window check: w 5 is rejected as a global default.** Summed F1 excess for
  DefenseFinder + TXSScan: 0.441 at w 3, 0.403 at w 5, 0.377 at w 8. The window stays a
  per-dataset setting: w 8 on the Spacedust collection (BGC regions of tens of genes on
  complete genomes), w 3 on DPANN (systems with a median span of 2-8 genes on short contigs).
  A window scaled to the target locus size or to contig length is the candidate rule.
- Modules: the largest component still holds 33,600-36,200 tokens at every window; the
  module definition is unchanged and still needs work.

## 2026-11-12 01:00 UTC — DPANN ESM++ small embedding queue

Jacob: embed DPANN with ESM++ small through a worker system of independent Slurm jobs that
pull from a queue, on srvp2020, DPANN only.

System: `benchmarks/redesign/embed_queue/` (README there), deployed at
`/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/`. Filesystem queue
with atomic-rename claims, heartbeats, stale-claim recovery, per-shard outputs through
temporary files, a self-check before any work, a per-shard float32 audit on the embedding
device, and a model-free finalize.

Run `runs/dpann__esmpp_small__20261111`:

- Source `ws_dpann/work/proteins_gid.faa`, 3,585,303 rows (row = gid).
- **2,547,528 unique sequences** (1,037,775 rows, 28.9%, repeat a sequence already seen);
  674,679,965 unique residues; 170 shards of about 4,000,000 tokens; lengths 9-24,483,
  median 205, 4,322 unique sequences longer than 2,046; 7 `J` residues mapped to `X`.
- Model `Synthyra/ESMplusplus_small` at revision `46c5f7d5` (17 June 2026). The current
  revision (`37b1a0f7`) declares torch >= 2.14 and transformers >= 5.17, which no
  environment on biotite provides; both revisions point at the same weights blob
  (`d099223765bc...`). Environment: the Talea venv (torch 2.7.1+cu126, transformers 4.57.6).
- Policy: bfloat16 autocast on GPU, float32 residue-only mean, whole protein in one pass.

CPU smoke test (job 77703, 24 threads, 5 min 35 s; 340 rows / 320 unique / 74,802 residues):
passed. Float32 single protein against padded batches: minimum cosine 0.9999999999998 over
32 proteins of 47-740 residues; finalize audit of 60 proteins: minimum cosine
0.99999999998; the 20 duplicate rows expanded correctly. CPU throughput 511 residues/s.
The comparison against native `transformers` ESM-C 300M did not run: the elsa environment
(transformers 5.5.0) has no `esmc` model type. Jacob objected to running the model on CPU;
the audit now runs inside the GPU workers and finalize loads no model.

Coordination with the Talea session (lanes 3-6 hold all four of Jacob's `gpu_h200_4`
running slots, with four queued successors): it agreed to exactly one H200 job from this
run. Pending jobs start in submission order on biotite (priority weights are 0), so that
job starts after the queued lane successors. The H200 wrapper therefore never submits
another job under `gpu_h200_4` (a refused successor submission ends a Talea lane chain at
the cap of 10 submitted jobs).

Submitted 00:55 UTC: 77717 on `-p gpu` (A5000, running, ascending length) and 77718 on
`-p gpu_h200 --qos=gpu_h200_4` (pending, descending length).

## 2026-11-12 01:20 UTC — Permissive sequence families match the embedding hybrids (jobs 77729-77733)

Jacob (01:10 UTC): once the DPANN embeddings are back, experiment freely toward the best
tool for this purpose.

Baselines queued while the embeddings run, all with the genome-level null.

Spacedust source207, conserved loci with gaps 0-2 / all locus kinds:

| Tokens | w | Conserved g0-2: F1 | Random | Excess [95% CI] | All kinds: F1 | Excess |
|---|---|---|---|---|---|---|
| MMseqs E<=1e-3, cov 50% (601,584 tokens) | 3 | 0.692 | 0.571 | +0.121 [+0.085, +0.156] | 0.754 | +0.064 |
| same | 5 | **0.758** | 0.576 | **+0.182** [+0.150, +0.213] | 0.798 | +0.098 |
| same | 8 | 0.745 | 0.561 | +0.184 [+0.152, +0.214] | 0.807 | +0.104 |
| hybrid ESM2-650M CPM r 0.05 on 30%/80% (334,660 tokens) | 5 | 0.737 | 0.576 | +0.161 [+0.127, +0.193] | 0.788 | +0.089 |
| same | 8 | 0.750 | 0.566 | +0.184 [+0.154, +0.213] | 0.801 | +0.099 |

**A sequence-only token set equals the embedding hybrid at every window.** The gain of the
hybrid arm over the strict 30%/80% clusters is reproduced by relaxing the MMseqs clustering
(sensitivity 7.5, E <= 1e-3, 50% bidirectional coverage, no identity floor). On this
benchmark the embedding contributes nothing that a sensitive sequence clustering lacks.
For reference: native Spacedust F1 0.676 (+0.065); calibrated ELSA chains 0.652-0.762 by
model; every-hit MMseqs anchors 0.727 (+0.073).

DPANN gold loci (instances with at least two members), F1 / control / excess:

| Tokens | w | Genes covered | DefenseFinder | TXSScan | CRISPRCasTyper |
|---|---|---|---|---|---|
| MMseqs 30%/80% (674,172) | 3 | 55.4% | 0.720 / 0.463 / +0.258 | 0.731 / 0.548 / +0.183 | 0.527 / 0.466 / +0.061 |
| MMseqs E<=1e-3, cov 50% (430,471) | 3 | 62.9% | 0.814 / 0.545 / +0.269 | 0.753 / 0.580 / +0.173 | 0.586 / 0.519 / +0.067 |
| same | 5 | 64.7% | 0.799 / 0.551 / +0.248 | 0.746 / 0.583 / +0.164 | 0.583 / 0.522 / +0.061 |

Broader families raise absolute F1 on DPANN (DefenseFinder 0.720 to 0.814) and coverage
(55.4% to 62.9% of genes); the control rises with them, so the excess is about level
(summed DefenseFinder + TXSScan +0.441 to +0.442). Window 3 again beats window 5 on DPANN.

Next, to locate what embeddings add, if anything:

1. Further breadth by sequence alone: MMseqs E <= 1e-3 with 30% coverage on both data sets
   (jobs 77754-77757).
2. Embedding merges on top of the permissive base, Spacedust, window 5: ESM2-650M and ESM-C
   300M genome-centred centroids, Leiden CPM at null p 1e-2, 1e-3 and r 0.05, 0.1, 0.2
   (hybrid jobs 77758, 77764; engine jobs 77759-77763, 77765-77769).
3. The same on DPANN with the ESM++ small vectors when they are finalized
   (`benchmarks/redesign/lab/embed_tokens.py`: unique-level kNN, graph communities, hybrids).

## 2026-11-12 01:40 UTC — Sequence-anchors agent: whitened-embedding arms (jobs 77597, 77598, 77617)

Full record: `benchmarks/redesign/ws_seqanchors/REPORT.md`, section "Whitened-embedding arms".

- **Verdict unchanged: embedding seeds do not materially extend sequence anchors on the
  Spacedust collection.** Over the union of the six sequence arms (9.82M accepted pairs,
  58,065 covered genes) the whitened hybrid adds 0.90% / 0.75% accepted pairs
  (ESM2-650M / ESM-C 600M; bar 2%) and 2.6% / 3.8% covered genes (bar 5%). The learned head
  on ESM-C 600M: 1.00% pairs, 2.7% genes.
- **The additions are real homologs.** 70.4% [69.0, 71.7] / 62.1% [60.7, 63.5] of the
  no-sequence-hit additions are structure-confirmed (68.1% for the learned head).
  Estimated confirmed homologs with no sequence hit: 44,780 (ESM2-650M), 32,836
  (ESM-C 600M), 47,615 (learned head), 0.3-0.5% of the sequence pairs.
- **Whitening helps embedding seeds on their own:** +26% / +64% accepted blocks over exact
  raw cosine, cross-phylum blocks 30,360 against 11,422 and 23,074 against 3,757; F1 excess
  +0.016 [+0.003, +0.032] / +0.012 [-0.009, +0.035] over raw. Whitened seeds alone still
  trail the sequence top-60 arm (-0.048 / -0.024).
- **In the hybrid the gain is zero:** whitened hybrid minus sequence top-60 is +0.001 /
  -0.002 in F1 excess, intervals containing zero.
- Exact raw top-60 contains 99.8% of the original HNSW seeds, so the whitened-versus-raw
  contrast isolates the metric.

Together with the token result above, two independent routes (pairwise anchors, family
tokens) give the same answer on this collection: a sensitive sequence search already
captures what the embeddings find, and the embedding-only remainder is real but small.

## 2026-11-12 01:45 UTC — Embedding merges on top of permissive sequence families (jobs 77758-77769)

Spacedust source207, window 5, genome-level null. Base = MMseqs E <= 1e-3, 50% coverage
(601,584 families). Families merged by genome-centred centroid cosine, mutual 10-NN, Leiden
CPM. Conserved loci with gaps 0-2 / all locus kinds:

| Tokens | Tokens (n) | Conserved g0-2: F1 | Random | Excess [95% CI] | Localized | All kinds: F1 | Excess |
|---|---|---|---|---|---|---|---|
| sequence base | 601,584 | 0.758 | 0.576 | +0.182 [+0.150, +0.213] | 0.768 | 0.798 | +0.098 |
| + ESM2-650M, null p 1e-3 | 513,458 | 0.772 | 0.581 | +0.191 [+0.159, +0.221] | 0.802 | 0.815 | +0.112 |
| + ESM2-650M, null p 1e-2 | 452,948 | 0.773 | 0.584 | +0.189 [+0.159, +0.217] | 0.797 | 0.811 | +0.108 |
| + ESM2-650M, CPM r 0.2 | 287,842 | 0.791 | 0.595 | +0.195 [+0.168, +0.222] | 0.831 | 0.821 | +0.116 |
| + ESM2-650M, CPM r 0.1 | 252,340 | 0.788 | 0.599 | +0.189 [+0.160, +0.218] | 0.826 | 0.824 | +0.120 |
| **+ ESM2-650M, CPM r 0.05** | 227,791 | **0.801** | 0.602 | **+0.199** [+0.172, +0.225] | 0.855 | **0.832** | **+0.127** |
| + ESM-C 300M, null p 1e-3 | 512,844 | 0.764 | 0.576 | +0.188 [+0.156, +0.219] | 0.778 | 0.805 | +0.105 |
| + ESM-C 300M, CPM r 0.1 | 273,296 | 0.777 | 0.593 | +0.184 [+0.153, +0.212] | 0.802 | 0.817 | +0.113 |
| + ESM-C 300M, CPM r 0.05 | 250,499 | 0.760 | 0.597 | +0.163 [+0.131, +0.194] | 0.797 | 0.816 | +0.111 |

Paired per-region differences in observed F1 (`engine/paired_compare.py`, 207 regions,
10,000 bootstrap resamples):

| A minus B | Conserved g0-2 | All kinds | Win / tie / loss |
|---|---|---|---|
| ESM2-650M r 0.05 on permissive base minus sequence base | +0.044 [+0.017, +0.070] | +0.034 [+0.014, +0.054] | 74 / 89 / 44 |
| ESM2-650M p 1e-3 minus sequence base | +0.015 [-0.001, +0.030] | +0.017 [+0.005, +0.029] | 31 / 156 / 20 |
| ESM-C 300M r 0.1 minus sequence base | +0.019 [-0.007, +0.044] | +0.019 [+0.000, +0.038] | 68 / 84 / 55 |
| ESM2-650M r 0.05 on permissive base minus the same on 30%/80% | +0.065 [+0.037, +0.092] | +0.044 [+0.024, +0.066] | 86 / 76 / 45 |
| ESM-C 300M r 0.1 minus ESM2-650M r 0.05 | -0.025 [-0.051, +0.000] | -0.015 [-0.034, +0.003] | 46 / 81 / 80 |

- **Embedding merges add F1 on top of the permissive sequence families**: +0.044 at the
  most aggressive ESM2-650M merge, with the interval above zero. This revises the reading of
  the previous entry: breadth by sequence explains the gain of the *old* hybrid over 30%/80%
  clusters, and a further embedding merge still helps.
- Random-placement F1 rises with the merge (0.576 to 0.602), so the excess gain is smaller
  (+0.182 to +0.199). The summaries held observed per-region F1 only; engine v1.2 also saves
  the per-region random F1, and the base and the two best hybrids are rerun at windows 5 and
  8 into `results_v3/` for a paired interval on the excess (jobs 77803-77808).
- ESM2-650M merges help more than ESM-C 300M merges. ESM++ small is the ESM-C 300M model, so
  the DPANN vectors are the weaker of the two for this step.

## 2026-11-12 01:55 UTC — Sequence breadth saturates at 50% coverage (jobs 77754-77757)

MMseqs E <= 1e-3 with 30% bidirectional coverage, against the 50% arm:

| Data set, window | Arm | Tokens | Result |
|---|---|---|---|
| Spacedust, w 5 | 50% coverage | 601,584 | conserved g0-2 F1 0.758, excess +0.182; all kinds 0.798, +0.098 |
| Spacedust, w 5 | 30% coverage | 524,727 | conserved g0-2 F1 0.763, excess +0.185 [+0.152, +0.216]; all kinds 0.809, +0.108 |
| DPANN, w 3 | 50% coverage | 430,471 | DefenseFinder 0.814 / 0.545 / +0.269; TXSScan 0.753 / 0.580 / +0.173; CRISPRCasTyper 0.586 / 0.519 / +0.067 |
| DPANN, w 3 | 30% coverage | 380,979 | DefenseFinder 0.824 / 0.549 / +0.275; TXSScan 0.727 / 0.584 / +0.144; CRISPRCasTyper 0.578 / 0.521 / +0.057 |

Relaxing coverage from 50% to 30% moves Spacedust F1 by +0.005 and lowers the summed
DefenseFinder + TXSScan excess on DPANN from +0.442 to +0.419. Further breadth from
sequence clustering parameters is exhausted; the +0.044 from the ESM2-650M merge is beyond
what the coverage relaxation gives. The 50% arm stays the sequence base.

## 2026-11-12 02:05 UTC — Paired excess for the embedding merges (jobs 77803-77808, `results_v3/`)

Engine v1.2 saves per-region random-placement F1, so the excess is paired as well. A = hybrid
on the permissive base, B = the permissive sequence base, same window; 207 regions.

| A | w | Selection | Observed F1: A, B, A - B [95% CI] | Excess A - B [95% CI] |
|---|---|---|---|---|
| ESM2-650M CPM r 0.05 | 5 | conserved g0-2 | 0.801, 0.758, +0.044 [+0.017, +0.070] | +0.017 [-0.009, +0.043] |
| ESM2-650M CPM r 0.05 | 5 | all kinds | 0.832, 0.798, +0.034 [+0.014, +0.055] | +0.029 [+0.009, +0.050] |
| ESM2-650M CPM r 0.05 | 8 | conserved g0-2 | 0.759, 0.745, +0.014 [-0.013, +0.041] | -0.019 [-0.046, +0.007] |
| ESM2-650M CPM r 0.05 | 8 | all kinds | 0.822, 0.807, +0.015 [-0.003, +0.032] | +0.009 [-0.009, +0.028] |
| ESM-C 300M CPM r 0.1 | 5 | conserved g0-2 | 0.777, 0.758, +0.019 [-0.007, +0.044] | +0.001 [-0.023, +0.027] |
| ESM-C 300M CPM r 0.1 | 5 | all kinds | 0.817, 0.798, +0.019 [+0.000, +0.038] | +0.015 [-0.004, +0.035] |
| ESM-C 300M CPM r 0.1 | 8 | conserved g0-2 | 0.733, 0.745, -0.012 [-0.036, +0.011] | -0.031 [-0.054, -0.007] |
| ESM-C 300M CPM r 0.1 | 8 | all kinds | 0.805, 0.807, -0.002 [-0.020, +0.017] | -0.003 [-0.021, +0.017] |

- **About 60% of the observed gain at window 5 is the random baseline rising.** For
  conserved loci the paired excess gain of the ESM2-650M merge is +0.017 with an interval
  that contains zero; for all locus kinds it is +0.029 [+0.009, +0.050].
- **At window 8 the merge gives nothing** (ESM2-650M) or costs excess (ESM-C 300M, -0.031).
  Merged tokens and a wide window both add support; together they over-extend loci.
- **ESM-C 300M merges give no excess gain at either window.**
- Best Spacedust configuration so far: permissive base + ESM2-650M CPM r 0.05 at window 5,
  conserved g0-2 F1 0.801 (excess +0.199), all kinds 0.832 (+0.127).

Reading for the DPANN run: ESM++ small is the ESM-C 300M model, the arm with zero paired
excess gain here. Predeclared for DPANN, before the vectors exist: an ESM++ small merge of
the permissive base counts as a gain if the summed DefenseFinder + TXSScan F1 excess at
window 3 rises by at least 0.02 over the base (+0.442) on held-out genomes (the half of
genomes with even `genome_code`; settings chosen on the odd half). Otherwise the DPANN
default stays sequence-only and the vectors are used for the kNN-recurrence and
context-merge experiments.
