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

## 2026-11-12 02:25 UTC — Engine v1.3: a window per token pair (predeclared test)

The window is the one engine setting that differs between the data sets (8 on the
Spacedust collection, 3 on DPANN; the predeclared global-window check failed). v1.3 adds
`--adaptive 1:2:3:5:8`: co-locations are collected at the largest window, each token pair
is scored at every window of the grid (genomes with the pair within w, against the Poisson
expectation for w), and the pair keeps the window with its smallest log P. Its instances
beyond that distance are dropped. Shuffled genomes go through the same selection, so the
empirical FDR accounts for the choice among five windows. `poisson_logsf` keeps log P finite
where the survival function underflows, so strongly conserved pairs are ranked too.

Predeclared: the adaptive window becomes the default if, with the permissive sequence base,
(a) on source207 its paired excess for conserved loci (gaps 0-2) is within 0.01 of the
better fixed window (5 or 8), and (b) on DPANN its summed DefenseFinder + TXSScan excess is
within 0.01 of window 3 (+0.442). Failing either, the window stays a per-dataset setting.

Also added: `engine/dpann_compare.py`, paired comparisons on the DPANN gold loci with a
genome-level bootstrap and a fixed split (tune = odd `genome_code`, test = even). On the
existing runs: permissive 50% against strict 30%/80% at w 3, DefenseFinder F1 +0.094
[+0.082, +0.106] but summed excess +0.001 [-0.010, +0.012]; w 3 against w 5 on the
permissive base, summed excess +0.030 [+0.024, +0.036]; 50% against 30% coverage, +0.023
[+0.014, +0.032].

## 2026-11-12 02:15 UTC — Adaptive window: the universal grid fails on DPANN (jobs 77826-77829)

(The heading of the previous entry, "02:25", ran ahead of the clock; this one is `date -u`.)

A = adaptive, B = fixed window, same tokens, paired.

| Data set, tokens | A | B | Observed F1 A, B | Excess A - B [95% CI] |
|---|---|---|---|---|
| Spacedust, permissive sequence base, conserved g0-2 | grid 1-8 | w 5 | 0.759, 0.758 | +0.011 [-0.010, +0.032] |
| same | grid 1-8 | w 8 | 0.759, 0.745 | +0.010 [-0.008, +0.029] |
| Spacedust, ESM2-650M merge r 0.05, conserved g0-2 | grid 1-8 | w 5 | 0.730, 0.801 | -0.035 [-0.063, -0.007] |
| same | grid 1-8 | w 8 | 0.730, 0.759 | -0.000 [-0.026, +0.025] |
| DPANN, permissive base, DefenseFinder + TXSScan summed | grid 1-8 | w 3 | | -0.045 [-0.054, -0.038] (test half -0.054) |
| same | grid 1-3 | w 3 | | **+0.018 [+0.013, +0.023]** (test half +0.018 [+0.012, +0.025]) |

Windows chosen for significant pairs with the 1-8 grid: Spacedust sequence base 16% / 13% /
13% / 20% / 38% at 1 / 2 / 3 / 5 / 8; DPANN 18% / 15% / 14% / 21% / 31%.

- **Predeclared criterion: (a) met, (b) failed.** With one grid for both data sets the
  adaptive window matches the best fixed window on the Spacedust sequence base and loses
  0.045 summed excess on DPANN. The largest window stays a per-dataset setting. The smallest
  log P favours wide windows for about a third of the pairs, and on DPANN's short contigs
  those pairs join neighbouring systems.
- **Within a cap, the per-pair window helps on DPANN**: grid 1-3 against fixed 3 gives
  +0.018 summed excess, on the held-out half too. New best DPANN configuration: permissive
  base, grid 1-3, summed excess +0.460 (DefenseFinder F1 0.795 / excess +0.281; TXSScan
  0.747 / +0.178).
- The window is a resolution setting (systems of a few genes against clusters of tens of
  genes), as in any clustering; the tool should expose it as "scale" with the per-pair
  selection inside it.
- Confound in the hybrid row: the coverage heuristic chose rule `P<=1e-6 & L>=2` for the
  adaptive run and `P<=1e-4 & L>=3` for the fixed runs. Controls: the hybrid with grid 1-8
  and the rule fixed to `P<=1e-4 & L>=3`, and both token sets with grid 1-5 against w 5.

## 2026-11-12 02:30 UTC — Adaptive window controls (jobs 77857-77859): not adopted

| Tokens (Spacedust, conserved g0-2) | A | B | Observed F1 A, B | Excess A - B [95% CI] |
|---|---|---|---|---|
| ESM2-650M merge | grid 1-8, rule fixed to `P<=1e-4 & L>=3` | w 5 | 0.775, 0.801 | -0.021 [-0.047, +0.004] |
| ESM2-650M merge | same | w 8 | 0.775, 0.759 | +0.013 [-0.009, +0.036] |
| permissive sequence base | grid 1-5 | w 5 | 0.748, 0.758 | -0.012 [-0.028, +0.003] |
| ESM2-650M merge | grid 1-5 | w 5 | 0.770, 0.801 | -0.026 [-0.051, -0.002] |

With the rule held fixed the adaptive hybrid run sits between the two fixed windows, so
part of the earlier deficit was the rule switch. Inside a cap of 5 the per-pair window
costs 0.012-0.026 excess on the Spacedust collection, where it gained 0.018 on DPANN inside
a cap of 3. Dropping a pair's co-locations beyond its most significant distance removes
support that cluster-scale loci need. **Verdict: `--adaptive` stays an option and is not
the default.** Fixed window per data set remains: 5 with merged tokens or 8 with sequence
tokens on the Spacedust collection, 3 on DPANN.

## 2026-11-12 02:45 UTC — Modules: communities replace connected components (engine v1.4)

A module is meant to name a recurrent neighbourhood type. The union-find definition
(components of pairs with J >= 0.5) put 26,832 of 33,888 module tokens into one component
on DPANN, so every locus carried the same label.

`lab/modules_lab.py` scores a module definition with no locus caller in between: each
curated instance with at least two members takes the plurality module of its members'
tokens; adjusted mutual information (AMI) with the caller's system type, homogeneity and
completeness. DPANN, permissive sequence base, window 3 (159,955 significant pairs, 40,919
tokens), DefenseFinder (2,472 instances, 158 types in the package):

| Definition (edge weight) | Modules | Largest (tokens) | AMI | Homogeneity | Completeness | AMI tune / test |
|---|---|---|---|---|---|---|
| union-find, J >= 0.5 (engine v1.1-1.3) | 2,458 | 26,832 | 0.28 | 0.21 | 0.67 | 0.272 / 0.264 |
| Leiden modularity r 5 (J) | 1,694 | 1,227 | 0.74 | 0.80 | 0.72 | 0.734 / 0.729 |
| Leiden modularity r 20 (J) | 2,042 | 245 | 0.75 | 0.90 | 0.67 | 0.750 / 0.738 |
| Leiden modularity r 5 (unweighted) | 1,621 | 1,349 | 0.72 | 0.73 | 0.74 | 0.704 / 0.716 |
| **Leiden modularity r 5 (J x log2 L)** | 1,692 | 1,243 | **0.80** | 0.86 | 0.77 | 0.795 / 0.798 |
| Leiden CPM r 0.02-0.4 (J) | 5,057-9,178 | 75-10 | 0.63-0.67 | 0.93-0.95 | 0.52-0.57 | |
| Louvain (networkx) r 5 (J x log2 L) | 1,699 | 1,224 | 0.76 | 0.78 | 0.78 | 0.765 / 0.759 |
| **Louvain (networkx) r 10 (J x log2 L)** | 1,833 | 701 | 0.79 | 0.89 | 0.73 | 0.791 / 0.778 |

On the strict 30%/80% clusters the best definition reaches AMI 0.66, so the permissive
families also give the better modules. TXSScan has a single system type among instances
with two or more members and CRISPRCasTyper labels are mostly `cas_putative`; both are
uninformative for this score (CRISPRCasTyper AMI 0.09-0.13 for every definition).

Engine v1.4: `--module-method louvain` (default; networkx Louvain, resolution 10, weight
J x log2 L, seed 17), chosen over Leiden because networkx is in every environment the
engine runs in while igraph is absent from the elsa environment; the cost is 0.02 AMI.
A significant pair whose two tokens fall in different modules names neither. Reruns:
77902 (DPANN, permissive base, w 3, with the gold-locus scorer) and 77903 (Spacedust,
permissive base, w 8).

## 2026-11-12 03:00 UTC — Modules on called loci (jobs 77902, 77903, 77919; engine v1.4 and v1.5)

DPANN, permissive sequence base, window 3. Loci and F1 are identical across engine
versions (932,123 conserved loci; DefenseFinder F1 0.814); only the module labels differ.
Score: module of each instance's best locus against its system type (`lab/show_modules.py`).

| Engine | Module of a locus | DefenseFinder best loci with a module | AMI | Hom. | Compl. | Plurality consistency (control) |
|---|---|---|---|---|---|---|
| v1.1 | union-find; plurality over boundary pairs | 2,225 / 2,472 | 0.241 | 0.19 | 0.54 | 0.903 (0.902) |
| v1.4 | Louvain; plurality over boundary pairs, cross-module pairs unlabelled | 2,047 / 2,472 | 0.795 | 0.89 | 0.74 | 0.725 (0.317) |
| **v1.5** | Louvain; plurality over the genes of the locus | **2,278 / 2,472** | 0.769 | 0.88 | 0.71 | 0.702 (0.253) |

v1.5 labels 92% of DefenseFinder best loci (83% in v1.4), 802 / 804 for TXSScan (402) and
3,423 / 3,820 for CRISPRCasTyper (2,176), at 0.026 lower AMI on the larger labelled set.
On the Spacedust collection (job 77903, window 8) v1.4 gives 1,797 modules, largest 1,397
tokens, with F1 unchanged (0.745 / 0.807). v1.5 is the engine default.

## 2026-11-12 04:33 UTC — Decisions from Jacob

- ESM2-650M embedding of DPANN is authorized. Submit it when the H200s are free (Talea
  lanes finished or talea-ae agrees); ESM++ small serves the first round of DPANN tests.
  The run needs an ESM2 backend in `embed_queue` (same shard queue, new model loader and
  pooling; 1,022-residue context needs a chunking rule for the 4,322+ long proteins).
- Framing concern: a tool positioned beside Spacedust (Steinegger lab) whose family step is
  MMseqs2 reads poorly as an independent alternative. Jacob may approach Martin Steinegger
  about joining. Consequence for the experiments: the embedding arms now carry the claim of
  independent value; a sequence-only result points toward a collaboration or an extension
  of their stack. A family step from another lab (for example DIAMOND DeepClust) is an
  untested option for the token table; the engine takes any gene-to-token table.
- Embedding status at 04:25 UTC: 166 / 170 shards done on job 77717; 77718 still pending.

## 2026-11-12 04:52 UTC — ESM++ small vectors for DPANN are final; round 1 queued; three workstreams launched

**Embedding (embed_queue run `dpann__esmpp_small__20261111`).** Finalized 20:33 PST 11 Nov by
job 78083. One RTX A5000 (job 77717) embedded all 170 shards in 12,895 s (52,319 residues/s);
the H200 worker 77718 never started and was cancelled by finalize. 2,547,528 unique sequences,
0 failed, 0 rows without a vector, 0 out-of-memory splits. Worker self-check: minimum cosine
0.99997 (gate 0.999). Shard audit, 510 proteins, float32 single-protein pass against the
stored bf16 batched vector: minimum 0.999975, median 0.999995. Assembly check: 0 of 20,000
rows differ. Products: `final/l2_f32__row.npy` (3,585,303 x 960), `final/mean_raw_f32__uid.npy`.

**Round 1 (jobs 78091-78108, registry `lab/emb1_jobs.tsv`, results `engine/results_dpann_emb1`).**
kNN of the unique sequences (zca10, k 50); hybrid merges of the permissive base e3c50 with
genome-centred centroids (CPM at null quantiles 0.5, 0.9, 0.99, 0.999 and at cosine 0.05,
0.1) and genome-centred zca10 centroids (the four quantiles); graph tokens from the mutual
20-NN graph (modularity 100, 300, 1000, 3000); the engine at window 3 on all 14 arms. The
criterion fixed on 12 Nov 00:57 UTC applies: a merge counts as a gain if the summed
DefenseFinder + TXSScan excess rises by at least 0.02 over the base (+0.442) on even genomes.

**Direction, after discussion with Jacob (six candidate ideas).**
1. Locus search and iterative locus profiles: go; "search with loci" is a core goal.
2. Synteny-supervised embedding head: check overlap with Tatta Bio first. Found: FlashPPI
   (Cornman, Tranzillo, Zulaybar, Bouzit, Hwang, PNAS 123(25), 2026; contrastive, initialised
   from gLM2 650M, partner retrieval plus residue contact maps; FlashPPI2 adds AlphaFold DB
   heterodimers; weights CC BY-NC) and Gaia (Science Advances 2025; gLM2_embed, contrastive on
   2.3 M AlphaFold DB structural clusters, context-aware retrieval over 131,744 genomes).
   Their positives are interaction partners and structural clusters; "same syntenic slot"
   positives are a different signal. Plan: keep it as the slot-emission model inside the locus
   profiles, with no standalone embedding claim. FlashPPI training positives were not stated
   on the pages read.
3. Genomic language model as a prior: parked until the H200s are free.
4. Sub-genic synteny: sizing measurement first (below).
5. Adjacency survival against a neutral clock: dropped. Jacob: gene-order decay cannot be
   modelled across divergences this deep; agreed, the clock has no dynamic range here.
6. Annotation-free truth by co-folding: Jacob's main idea for the paper.

**Workstreams (Opus agents; code under `benchmarks/redesign/<name>/`, each with REPORT.md).**
- `locus_search`: headroom of same-type gold instances beyond tokens; retrieval benchmark
  (token, embedding, combined, iterative profile) with a co-localization null; predeclared:
  embeddings count as a gain if mean AP on even-genome DefenseFinder queries rises by at least
  0.02 over tokens (paired bootstrap CI excluding 0) at no more than 1.5x the unlabelled-hit
  rate; then a profile library built on odd genomes and scanned on even genomes.
- `cofold`: unannotated 2-4 gene blocks from engine v1.5, ranked by divergence spanned,
  operon geometry and foldability; ESMFold2 (Biohub API) hetero-complexes for about 150
  candidates in two distant instances each, with non-cognate, uncalled-neighbour and positive
  controls; criteria for "obviously co-folding" written before folding.
- `subgenic`: split-in-synteny composites, fragments, token loss, impact on gold loci, giant
  proteins. Predeclared gate: build sub-genic tokens for DPANN if composites involve at least
  1% of genes or affect at least 2% of gold instances with two or more members.

ESM2-650M on DPANN stays queued behind the Talea scan lanes (about a week, Talea agent's
estimate); nothing of ours is submitted to gpu_h200.

## 2026-11-12 05:45 UTC — Round 1 hybrid merges on DPANN fail the predeclared criterion; co-fold screen 1

**Hybrid merges of the permissive base with ESM++ small centroids (jobs 78092-78103, window 3,
genome-level null; `engine/results_dpann_emb1/compare_vs_base.txt`).** Paired genome bootstrap
against `results_dpann_v2/dpann__mmseqs__seq__e3c50s75__w3`; A - B, 95% interval.

| Arm (centroids, CPM resolution) | Tokens | DefenseFinder F1, test | Summed DF+TXSS excess, tune | Summed excess, test |
|---|---|---|---|---|
| base e3c50 | 430,471 | 0.806 | +0.436 | +0.447 |
| genome-centred, q0.5 | 301,015 | +0.012 [+0.003, +0.020] | -0.015 [-0.025, -0.005] | -0.008 [-0.017, +0.002] |
| genome-centred, q0.9 | 319,490 | +0.009 [+0.003, +0.015] | -0.022 [-0.030, -0.013] | -0.016 [-0.025, -0.009] |
| genome-centred, q0.99 | 347,364 | +0.003 [-0.001, +0.007] | -0.016 [-0.023, -0.010] | -0.008 [-0.014, -0.001] |
| genome-centred, q0.999 | 375,007 | +0.003 [-0.000, +0.006] | -0.007 [-0.011, -0.003] | -0.002 [-0.007, +0.003] |
| genome-centred, r0.05 | 204,494 | +0.012 [+0.001, +0.024] | -0.012 [-0.023, +0.000] | -0.021 [-0.033, -0.009] |
| genome-centred, r0.1 | 218,913 | +0.020 [+0.009, +0.031] | -0.019 [-0.030, -0.008] | -0.013 [-0.025, -0.002] |
| genome-centred zca10, q0.5 | 94,450 | +0.051 [+0.037, +0.066] | -0.019 [-0.035, -0.005] | -0.020 [-0.036, -0.006] |
| genome-centred zca10, q0.9 | 166,703 | +0.038 [+0.026, +0.050] | -0.031 [-0.043, -0.018] | -0.032 [-0.044, -0.019] |
| genome-centred zca10, q0.99 | 226,258 | +0.036 [+0.025, +0.048] | -0.005 [-0.015, +0.006] | +0.000 [-0.012, +0.012] |
| genome-centred zca10, q0.999 | 263,018 | +0.032 [+0.022, +0.042] | -0.010 [-0.020, +0.002] | -0.004 [-0.015, +0.007] |

**Outcome of the predeclared test: no arm gains.** The criterion was a rise of at least 0.02
in summed DefenseFinder + TXSScan excess on even genomes; the best arm chosen on odd genomes
(zca10 centroids, q0.99) gives +0.000 [-0.012, +0.012]. Whitened centroids raise observed
DefenseFinder F1 by 0.032-0.051 and the random-placement control by the same amount: the
merged families lengthen loci without localizing systems better. The DPANN default stays the
sequence-only permissive base. The kNN job 78091 (zca10, k 50; needed for the graph arms
78104-78108) was still running at 64 threads after one hour.

**Co-fold screen 1 (workstream `cofold`; `benchmarks/redesign/cofold/REPORT.md`).** 150
unannotated two-gene blocks called by the engine, two distant instances each, ESMFold2 through
the Biohub API (1,047 folds), criteria written 12 s before the batch. Pass rate on the three
structural criteria: engine-called pairs 27 / 300 (9.0%), non-cognate partner swaps 0 / 300,
adjacent dark pairs the engine did not call 4 / 297 (1.35%; odds ratio 7.2, p = 2.2e-5),
mates inside curated systems 19 / 150 (12.7%). Eight candidates pass in both replicates with a
cognate-minus-non-cognate margin of at least 0.30; one is refuted as a split gene (fused
homologs in 27% as many genomes), one is flagged for disorder zipping, six remain. Several of
the six have homologs carrying Pfam families of two-gene addiction modules, so screen 2
(workstream `cofold2`, running) adds a profile and structure triage before folding and extends
to blocks of three and four genes. Protenix v2 confirmation waits for a GPU with more than the
12 GB host memory the `gpu` partition QOS allows per job.

## 2026-11-12 06:02 UTC — Round 1 complete: embedding-only graph families on DPANN (jobs 78091, 78104-78108)

kNN of the 2,547,528 unique ESM++ small vectors (zca10, IVF 4,096 lists, 48 probes, k 50) took
4,519 s at 64 threads; neighbour cosine q10 / q50 / q90 is 0.602 / 0.864 / 0.991 at rank 1 and
0.426 / 0.653 / 0.846 at rank 50, against a random-pair q99 of 0.14 and q99.9 of 0.32. The
search is the slow step (dense families make unbalanced lists); an HNSW or a progress log is
needed before the next kNN at this scale.

Graph tokens = Leiden modularity communities of the mutual 20-NN graph, no sequence clustering.
Engine at window 3, paired against the permissive sequence base (`compare_graph_vs_base.txt`),
even genomes, A - B with 95% interval:

| Resolution | Tokens | DefenseFinder F1 | DefenseFinder excess | TXSScan excess | CRISPRCasTyper excess | Summed DF+TXSS excess |
|---|---|---|---|---|---|---|
| 100 | 113,638 | -0.021 [-0.038, -0.004] | -0.049 [-0.064, -0.033] | -0.081 [-0.092, -0.068] | -0.021 [-0.030, -0.013] | -0.129 [-0.148, -0.110] |
| 300 | 120,083 | +0.006 [-0.009, +0.021] | -0.005 [-0.020, +0.009] | -0.089 [-0.101, -0.078] | -0.026 [-0.035, -0.016] | -0.094 [-0.112, -0.076] |
| 1000 | 132,627 | +0.003 [-0.012, +0.018] | +0.022 [+0.008, +0.037] | -0.095 [-0.108, -0.083] | -0.024 [-0.033, -0.014] | -0.073 [-0.093, -0.054] |
| 3000 | 153,520 | +0.001 [-0.013, +0.016] | +0.044 [+0.029, +0.059] | -0.074 [-0.085, -0.062] | -0.029 [-0.039, -0.019] | -0.029 [-0.047, -0.011] |

Embedding-only families fail the summed criterion at every resolution. Two facts worth
keeping: (1) with no alignment at all they equal the sequence base on DefenseFinder F1 (0.807
against 0.806) and, at resolution 3000, localize those systems better (excess +0.313 against
+0.268); (2) they lose 0.12 of TXSScan F1 at every resolution, so the loss is specific to
that class of proteins and independent of community size. Untested reading: secretion and
pilus components are short, hydrophobic, low-complexity proteins whose mean-pooled vectors
group by composition. Test: TXSScan member genes' token purity against hydrophobic fraction
and length, compared with DefenseFinder members.

Round 1 verdict on DPANN: ESM++ small vectors add nothing as a merger of sequence families and
are worse than sequence families as a stand-alone token source on the predeclared criterion.

## 2026-11-12 06:41 UTC — Speed: HNSW kNN validated on the DPANN vectors (job 78277); wrapper changes

Jacob asked for speed wherever possible. Changes (commit `e525b9a`):
- `embed_tokens.py knn --index hnsw` (FAISS HNSW, M 32, efConstruction 200, efSearch 128),
  progress stamps, `--exact-n` (recall against brute force on sampled queries) and
  `--check-npz` (scores an earlier kNN file on the same sample).
- `run_recurrence_dpann_arm.sbatch` runs the two scorer passes side by side and writes
  `compare_vs_base.txt` (paired genome bootstrap against the permissive base) in the job.
- The three workstream agents may run four jobs each.

Validation, 2,547,528 ESM++ small vectors, zca10, k 50, 64 threads, 5,000 exact queries:

| Index | kNN seconds | recall@1 | recall@10 | recall@50 |
|---|---|---|---|---|
| IVF flat, 4,096 lists, 48 probes (job 78091) | 4,436 | 0.9952 | 0.9933 | 0.9903 |
| HNSW (job 78277) | 1,302 (build about 810, search 491) | 0.9924 | 0.9931 | 0.9910 |

HNSW is 3.4 times faster at the same recall and is now the wrapper default
(`knn__<tag>__<transform>__k<k>__hnsw.npz`). Brute force on 5,000 queries took 45 s, so an
exact all-against-all pass would cost about 6.4 h on 64 threads.

## 2026-11-12 07:09 UTC — Sub-genic sizing on DPANN (workstream `subgenic`); co-fold triage; length effect

**Sub-genic sizing** (`benchmarks/redesign/subgenic/REPORT.md`; gate fixed 04:59 UTC before any
measurement). MMseqs2 local alignments of 371,905 query sequences (10% of genomes plus every
gold neighbourhood) and all 4,330 giants against the 2,547,528 unique sequences.
- Split side of an adjacent split/fused relation: 7.8% of genes; fused side 25.1% (grows with
  collection size). Adjacent groups are 49.7x a within-genome shuffle; 91.3% co-oriented and in
  the order of the intact gene. The literal predeclared control (pieces more than 3 genes apart)
  exceeds the adjacent rate, so gate (a) fails as written; it is not a chance null.
- Gate (b) met: 5,006 / 7,096 gold instances contain a composite, split or fragment member
  (fragment alone 7.5%, split side alone 14.6%).
- Whole-gene tokens already absorb composites (0.1% lost; 85.1% share a token with a piece) and
  composite-containing instances recover better (odds ratio of poor recovery 0.53).
- Fragments: 4.2% of genes; 39.2% of contig-edge genes against 1.7% interior; 27.1% lost from
  tokens (baseline 1.7%); rate tracks N50 (rho -0.80); gold instances with a fragment member
  recover worse (odds ratio 2.50, F1 -0.088).
- Split genes are clade-wide states: per-genome rate 5.2 / 7.4 / 10.5% (q10 / q50 / q90),
  unrelated to completeness or N50 (|rho| <= 0.05); 59% of split loci lie in families where the
  split form is the majority; 47.1% in families split in >= 80% of an order's genomes.
- Giants (>= 2,046 aa): 99.4% have a local homolog in another genome; 57% are tiled by two or
  more adjacent smaller genes elsewhere (1,104 tilings broken-call-like, 687 separate genes, 680
  mixed); 68.6% carry internal repeats (length control 30.6%).
- Repair estimate: merging broken-call-like split loci about 80,200 genes (2.2%); a
  shorter-sequence coverage rule rejoins about 90,600 fragments (2.5%); 574 gold instances
  touched (8.1%).
Decision: residue-level sub-genic tokens are parked for DPANN; the cheap repair goes to an
engine test (workstream `repair`).

**Co-fold triage (workstream `cofold2`, part A; `benchmarks/redesign/cofold/partA_triage.tsv`).**
HHblits/HHsearch against Pfam 39 and PDB100, RPS-BLAST against CDD, hmmsearch Pfam GA, Foldseek
against PDB and AlphaFold Swiss-Prot: nine of the ten screen-1 dossier blocks have a chain that
resembles a characterised family (ribbon-helix-helix + PIN, ParD/ParE-like, MazE-like,
YoeB-like, ferritin-like pieces, a retron-effector-like protein with a helix-turn-helix
partner); rank 1 (60 + 70 aa; 249 genomes, 21 orders, 7 phyla) has no similarity at the
predeclared bar. All observations; no curated caller emitted a system. The shifted-frame
overlaps of ranks 27 and 106 are two translationally coupled genes (own start codon in every
instance, no shift motif, no fused homolog). The four high-scoring uncalled pairs are confined
to one genus each and fail the engine's two-lineage rule. Jacob dropped Protenix confirmation.

**Length effect in screen 1.** The ranking has no length term (Spearman of orders spanned
against summed length -0.05; folded set median 406 aa against 421 aa in the pool). The strict
pass rate falls with summed length for engine-called pairs (40% under 200 aa, n = 20; 10.7% at
200-300; 7.0% at 300-450; 4.8% above 450, n = 124) and for curated mates (33%, 20%, 16.7%,
10.6%); partner swaps pass 0 / 20 under 200 aa. Screen 2 folds blocks of 450 aa or more with and
without the block MSA and reports by length class.

## 2026-11-12 07:32 UTC — Co-fold screen 2 (workstream `cofold2`; `benchmarks/redesign/cofold/REPORT_screen2.txt`)

Pool widened to windows of up to 8 genes, proteins of 60-1,200 aa, blocks up to 2,500 aa: 392
block types pass every filter; 61 screened (42 multi-gene, 19 pairs with a protein over 700 aa).
Triage before folding (HHblits/HHsearch against Pfam 39 and PDB100, RPS-BLAST against CDD,
Pfam GA, later Foldseek) removed 37 as resembling characterised families (16 of the 19 long
pairs; PDB100 HHsearch flagged 54 of 62 characterised families). 24 blocks folded (18 of three
genes, 3 of four, 3 pairs; 472 Biohub items in total).

| Class | Folds with a passing interface | Rate |
|---|---|---|
| Engine-called blocks | 12 / 42 | 28.6% |
| Curated positives | 17 / 56 | 30.4% |
| Uncalled windows | 3 / 42 | 7.1% |
| Partner swaps | 3 / 90 | 3.3% |

Engine-called against uncalled: odds ratio 5.2, p = 0.02; against swaps: 11.6, p = 6.5e-5. Ten of
the 12 passing folds come from blocks with Pfam DUF7287 / DUF7288 families. Block-paired MSAs
(median depth 12) never raise the pass rate at 450-800 aa (10 / 31 single sequence, 8 / 31 with
MSA) or above 800 aa (1 / 10 both), so shallow MSAs do not rescue larger complexes.

No screen-2 block has every chain free of detectable similarity. Shortlist: (1) s2 rank 34, a
four-gene cassette in Nanobdellota (10 genomes, 2 orders): interfaces AB, AD, BC, BD pass in
both replicates (pair ipTM 0.79-0.86, controls 0.07-0.49); chain A has a prepilin-type signal
peptide and resembles a minor pilin, chains B and C are DUF7287 / DUF7288; every chain has a
hydrophobic N-terminal helix carrying 0-52% of the interface. (2) s2 rank 7, the same DUF
architecture in Micrarchaeota and Iainarchaeota (28 genomes, 3 orders): one interface passes
single-sequence (0.84 / 0.82) and fails with the MSA (interface PAE 5.5 / 6.3 A). (3) Screen-1
rank 1 (60 + 70 aa; 249 genomes, 21 orders, 7 phyla): no detectable similarity in any search,
ipTM 0.856 / 0.708, fused 133-139 aa homologs in 3 genomes. Follow-up workstream `cofold3`
examines rank 1 and the DUF cassette in depth.

## 2026-11-12 07:46 UTC — Locus search on DPANN (workstream `locus_search`; `benchmarks/redesign/locus_search/REPORT.md`)

Query = one gold instance, database = all genes of all other genomes; hits are windows scored
by per-slot evidence with a circular-shift co-localization null (200 replicates). DefenseFinder
queries with >= 2 members on even genomes (1,247); parameters tuned on odd genomes.

| Method | MAP | Strict MAP | Recall E<=0.01 | Unlabelled hits per query E<=0.01 |
|---|---|---|---|---|
| M0 tokens | 0.370 | 0.256 | 0.094 | 0.72 |
| M1 embedding rank | 0.442 | 0.257 | 0.093 | 2.59 |
| M2 tokens + 0.5 x embedding rank | 0.576 | 0.362 | 0.115 | 0.92 |
| M3 iterative profile (grow, round 3; selected on odd) | 0.615 | 0.399 | 0.174 | 2.81 |
| S1 MMseqs2 bitscore rank (control) | 0.602 | 0.389 | 0.170 | 4.47 |
| S2 tokens + sequence rank | 0.620 | 0.407 | 0.180 | 4.44 |
| S3 = S2 + 0.5 x embedding rank | 0.698 | 0.454 | 0.182 | 4.51 |

- Predeclared criterion (best of M1-M3 selected on odd genomes gains >= 0.02 AP over M0 at no
  more than 1.5x the unlabelled rate): not met; the selected M3 gains +0.245 [+0.235, +0.255]
  at 3.9x. M2 would pass (+0.206 [+0.198, +0.213] at 1.27x) and was not the selected method.
- Graded per-gene evidence is the main lever (tokens 0.370; sequence ranks 0.602-0.620), and
  sequence search supplies it better than ESM++ small (S2 - M2 = +0.044 [0.036, 0.052]).
- First controlled gain for the embeddings: S3 - S2 = +0.079 AP [+0.074, +0.083] (strict
  +0.047) with +0.08 unlabelled hits per query; recall at E <= 0.01 +0.002, so the embedding
  reorders the deeper list. This comparison was added after the predeclaration.
- Token-invisible targets: 120,946 even-genome pairs are embedding-reachable in phase 0; M1
  retrieves 2.5% of them at E <= 0.01.
- Profile library built on odd genomes and scanned on even genomes: DefenseFinder F1 0.675
  against 0.806 for the engine, excess +0.061 [+0.045, +0.078], coverage 0.773 against 0.906;
  298 genomes per second.
Follow-up workstream `locus_search2`: what the unlabelled hits are, a rate-constrained
selection, a fitted-tail null, and a token-pair library.

## 2026-11-12 07:56 UTC — Spec to Sharur; one-kernel hypothesis and its first test

- `docs/redesign/SPEC_family_remote_homology.md` (CDD per protein; MMseqs2 families, HHblits
  enrichment, HHsearch against HH-format Pfam 39 and PDB100; schema, cost, validation set) was
  sent to the Sharur session at Jacob's request. Measured cost: 173 families through all searches
  in 82 min on 24 CPUs (job 78254).
- Jacob's objection to the locus-search result: three similarity inputs make a hodgepodge and
  slow the method; the task is a clustering task. Working hypothesis in reply: tokens, sequence
  ranks and embedding ranks are feature maps for one kernel,
  K(a, a') = k(x_a, x_a') * sum_delta k(x_{a+delta}, x_{a'+delta'}); the recurrence engine is
  this kernel with one-hot tokens (token-pair recurrence = collisions of gene bigrams). With a
  dot-product k the neighbourhood term is c_a . c_a' with c_a the summed neighbour vectors, so K
  is the Gram matrix of x_a (x) c_a and nearest-neighbour search plus one graph clustering would
  give context-aware families, loci and search from a single input. A learned head that distils
  alignment evidence and context into one vector is the place for the language model.
- Workstream `context_kernel` tests the cheapest form: ESM++ small kNN edges weighted by
  neighbourhood agreement before clustering, then the engine. Predeclared: on even genomes the
  best arm (chosen on odd) has TXSScan excess >= base - 0.04 (context-free graph arms: -0.074
  to -0.095), summed DefenseFinder + TXSScan excess >= base - 0.01, and at the edge level
  AUROC(similarity x context) - AUROC(similarity) >= 0.05 for same-type against different-type
  gold pairs.

## 2026-11-12 08:15 UTC — Deep dive on the two remaining co-fold candidates (workstream `cofold3`; `benchmarks/redesign/cofold/REPORT_deepdive.txt`)

**Candidate 1 (tokens 81718 and 393054) is the start of a three-gene operon.** A (about 60 aa)
and B (about 70 aa) have no detectable similarity in any search (HHblits 3 iterations against
UniRef30, HHsearch against Pfam 39 and PDB100, hmmsearch against UniRef50, Foldseek against PDB
and AlphaFold Swiss-Prot). The gene directly downstream of B in 258 of 267 instances
(co-oriented in all 258, overlapping in 168) is C (token 130016 in 240; 214 aa), whose family
resembles CD-clan (caspase-like / CHAT) cysteine proteases (Pfam DUF26930 99.6%, Peptidase_C25
95.3%, PDB caspase-3 94.7%; Foldseek SAVED-CHAT 8tl0 E 7e-8). DefenseFinder emitted a curated
Borvo call (component BovA) on C in 8 of 268 loci; no curated call on any A or B.
- Folds, 12 instances over 12 orders and 7 phyla: A+B ipTM 0.653-0.937 (median 0.784), partner
  swaps median 0.125 (max 0.473), cognate minus best swap >= 0.41 in 12 / 12; 1:1 heterodimer
  (A2B2, A2B, AB2 add no interface). A+B+C ipTM 0.856-0.953 in 11 / 12 (median 0.905; the 12th
  has a 104-aa C, 0.383); B-C interface passes in 11 / 12 (70-93 contacts, PAE 1.2-3.8 A); swaps
  of C: median 0.29; B swaps against C: median 0.18.
- C carries a His (98% of 258 rows) and a Cys (96%) 34 residues apart; B contacts both in
  11 / 11 models (CB distance 3.7-6.6 A), A lies 25-32 A away.
- Fused A-B single ORFs: 3 in DPANN, 2 bacterial UniProt entries (A0A661IN52, A0A2W7RED9); their
  models match the A+B arrangement (TM 0.68-0.82; a single half reaches 0.36-0.51).
- Distribution: 249 genomes (HMM pairs in 295), 4-21% of genomes per phylum, flat across
  completeness bins; adjacent and co-oriented in 325 / 325 HMM pairs, ATGA overlap in 207.
  UniRef homologs of A|B are mainly Patescibacteria (18 of 26 bacterial hits at P >= 95).
- Context: curated DefenseFinder members in 8.8% of windows against 2.5% of length-matched
  controls on the same contigs (p 0.0027). Engine module 1167 holds A, B, the C tokens and
  radical-SAM families.
- No coevolution evidence: MSA folds equal single-sequence folds; interface columns are no more
  conserved than others (p 0.81).
Reading: two unrecognised small proteins form a conserved heterodimer that packs, through B,
onto the His-Cys site of a caspase-like protein in a defence-associated operon. Function
UNVERIFIED. View script (not opened; Jacob asked for no GPU use on the laptop):
`benchmarks/redesign/cofold/view_candidate1_abc.cxc`.

**Candidate 2 (DUF7287 / DUF7288 cassette) fails as a showcase.** Recognised (those Pfam
families by HHsearch; TXSScan component arCOG06117 on homologs of one chain; minor-pilin-like by
Foldseek); 1-2 genes from curated Archaeal-T4P members in 8 / 8 instances whose contig carries
a call; isolated pairs pass in 1 / 18; truncating the N-terminal helix keeps the interfaces in
one replicate per block. Pilin-like control pairs stay low (72 pairs, median ipTM 0.18, none
>= 0.75). No overlap with the earlier cluster:1043 or cluster:3913 leads.

## 2026-11-12 08:52 UTC — Locus search round 2 (workstream `locus_search2`; `benchmarks/redesign/locus_search/REPORT2.txt`)

Predeclaration `results/criterion2.json` written 07:58 UTC before any round-2 number. Reruns of
round-1 configurations reproduce round 1 exactly.

- **Confirmatory test of the embedding's added value: met.** DefenseFinder, 5-fold genome
  cross-validation (2,447 queries): S3 - S2 = +0.077 AP [+0.074, +0.080] (strict +0.046); every
  fold chose the round-1 configurations. Frozen configurations on the other callers: TXSScan
  +0.260 [+0.250, +0.269], CRISPRCasTyper +0.013 [+0.012, +0.015].
- **Controls on that gain.** With each gene's embedding list swapped for another gene's, S3 - S2
  on DefenseFinder falls to +0.012; true minus shuffled is +0.064 [+0.061, +0.068], 84% of the
  gain. CRISPRCasTyper: true minus shuffled +0.002 (the gain is non-specific there). TXSScan: at
  matched floor (tau 3) the gain is +0.039; the rest is a floor effect in a one-type caller.
- **Unlabelled hits carry the query type's domains.** At E <= 0.01, 92.6-96.5% of unlabelled
  hits have a raw DefenseFinder profile hit used by curated calls of the query's type
  (span-matched windows on the same contigs: 0.9-2.2%); 97.7-100% meet the predeclared
  "supported" rule. Unsupported unlabelled hits per query: M0 0.000, M2 0.002, S2 0.028, S3
  0.029, M3 grow round 3 0.064. AbiE queries give 54% of S3's unlabelled hits. Observed domains
  only; no system is named for an unlabelled hit.
- **Rate-constrained selection.** Counting every unlabelled hit against the rate: S2 with
  sequence weight 0.75 (AP 0.619, +0.249 [+0.238, +0.258] over M0 at 1.53x). With supported hits
  exempt: S3 (AP 0.698, +0.328 [+0.317, +0.339]; 0.018 unsupported per query).
- **Fast null.** Gumbel fit to 40 replicate maxima plus a per-method calibration curve learned on
  other genomes: observed / expected 1.00 [0.96, 1.05] at E = 0.01, 1.00 at 0.1, 1.02 at 1, 1.96
  at 0.001. Uncalibrated small-sample tails are 1.4-2.2x anti-conservative at 0.01, and the
  200-replicate empirical E is at its resolution limit there. Null cost falls 5x (S3: 10.4 to
  2.1 CPU-s per query).
- **Interactive search.** `locus_search/code/interactive.py search --gids ...`: median 4.8 s per
  query in a session on 24 CPUs (embedding 0.44 s, indexed MMseqs2 3.22 s, scan 0.03 s, null
  0.92 s). Deep lists carry AP: cutting embedding ranks from 20,000 to 2,000 costs 0.034 AP
  whether exact or HNSW; a faster MMseqs2 setting costs 0.033; both 0.078-0.086.
- **Pair library.** Odd-genome significant token pairs applied to even genomes with no statistic
  recomputed (1.8 M genes in 11.5 s): DefenseFinder F1 0.721 (engine 0.806, module library
  0.675) at the engine's excess (0.267 against 0.268); TXSScan F1 0.752 against 0.758.
- Spacedust replication: source207 carries no type label; nothing run.

## 2026-11-12 09:12 UTC — Candidate 1: independent tests (workstream `cofold4`; `benchmarks/redesign/cofold/REPORT_candidate1.txt`)

The A-B-C unit (A about 60 aa, B about 70 aa, both unrecognised; C 214 aa, caspase-like) holds
outside DPANN and B occupies C's active-site groove.
- **Outside DPANN, 40 public loci found by accession lookup (39 bacterial, 1 Bathyarchaeota):**
  37 have a C-family gene within 3 genes against 0 of 51 length-matched control windows on the
  same records (p 2.3e-22); in 35 of the 37, C is the co-oriented gene directly after B. 36 are
  CPR/Patescibacteria; the rest Omnitrophota, Myxococcota, Bacteroidota, Bathyarchaeota. The
  bacterial C copies keep the His and Cys in 36 of 38. No other neighbour recurs.
- **Inside DPANN:** of 309 high-scoring C genes with an upstream gene, 306 have A or B directly
  upstream (293 both). 118 DefenseFinder Borvo calls exist; 10 fall on C-family proteins and the
  other 108 on 51 unrelated families (ribosomal proteins L22 and S13, prolyl-tRNA synthetase
  among them), so that call has low specificity in this dataset.
- **Structure (11 passing models):** relative accessibility of C's His / Cys is 0.25 / 0.19 alone
  and 0.07 / 0.02 in the complex; B buries a median 71% / 86% of their side-chain surface. B's
  first hairpin strand pairs with a C strand; B's invariant Gly (99.6%) touches the Cys in 10 of
  11 models. C's His and Cys superpose on the annotated catalytic dyad in 66 of 66 alignments to
  six protease entries (gingipain R 1cvr: Cb 0.75 and 1.05 A away). B covers 79% of the 1cvr
  inhibitor's atoms within 3 A, running in the same direction (cosine 0.95-0.99).
- **Covariation, B-C contact against non-contact column pairs (258 paired rows, effective 174.6):**
  AUC 0.697 (MI with average-product correction) and 0.763 (mean-field DCA); pairing-shuffle
  nulls 0.45-0.56 (global, within phylum, within order; p <= 0.01). Within-genus shuffle: MI not
  separable (p 0.18), DCA p 0.0099. A-B shows the same pattern (0.716 / 0.766). This revises the
  deep dive's "no coevolution evidence".
- **Junctions:** A-B is translational coupling (own ATG in 314 of 325; ATGA in 206). -1 slippery
  sites are enriched at A-B junctions against ATGA junctions on the same contigs (odds ratio
  3.06, p 7.5e-11), almost all in Nanobdellota (44 of 175 pairs; 0 of 67 in Micrarchaeota). B-C:
  no enrichment.
- **Bacterial folds (60 items):** CPR A+B+C ipTM median 0.910 (B-C passes in 10 of 11); B+C
  passes in 11 of 13 including Bathyarchaeota (0.920) and Omnitrophota (0.836); every partner
  swap fails (0 of 18).
Two readings, both UNVERIFIED: B is a substrate-mimicking inhibitor of C, or B is cleaved by C
after the Gly. Biohub items used across all co-fold work: 2,187.

## 2026-11-12 09:16 UTC — Context-weighted similarity (workstream `context_kernel`; `benchmarks/redesign/context_kernel/REPORT.txt`)

Single input: ESM++ small (zca10) kNN expanded to gene-level cross-genome edges (164.9 M
undirected, 64 partners per gene), neighbourhood agreement per edge (dense c_a . c_a' and
best-match forms, windows 1, 3, 5), support at 1% FDR against a random-partner null, Leiden
families, engine at window 3. Predeclared 08:07 UTC.

| Arm | Tokens | DefenseFinder F1 | TXSScan F1 | TXSScan excess vs base | Summed DF+TXSS excess vs base (even) |
|---|---|---|---|---|---|
| MMseqs2 base | 430,471 | 0.806 | 0.758 | 0 | 0 |
| Context gate (dense w1), weight sim x ctx (chosen on odd) | 880,124 | 0.741 | 0.692 | -0.013 [-0.024, -0.001] | -0.004 [-0.022, +0.016] |
| Same supported edges, weight sim | 879,919 | 0.744 | 0.682 | -0.023 | -0.014 [-0.032, +0.005] |
| All edges, weight sim x ctx, r11000 | 177,500 | 0.841 | 0.608 | -0.114 | -0.061 [-0.082, -0.041] |
| All edges, weight sim (V0), r19000 | 297,708 | 0.775 | 0.632 | -0.044 | +0.038 [+0.017, +0.062] |
| Gate built on permuted gene order, weight sim | 677,277 | 0.438 | 0.346 | -0.074 | -0.094 [-0.127, -0.060] |
| Strict gate (best-match w5), weight sim x ctx | 1,632,458 | 0.060 | 0.585 | +0.043 | -0.269 [-0.295, -0.245] |

- P1 (TXSScan excess >= base - 0.04): met. P2 (summed excess >= base - 0.01): met, competitive.
  P3 (edge AUROC of sim x ctx exceeds sim by 0.05 for same-type against different-type gold
  pairs): not met, -0.353 [-0.371, -0.335]; context alone 0.37-0.50 against 0.841 for similarity.
- **The multiplicative kernel is refuted as a weighting.** The gain comes from using context as
  an edge filter under the true gene order; weighting the same edges by context adds nothing,
  and weighting all edges by sim x ctx costs TXSScan 0.11-0.21 of excess.
- **A strict context filter removes DefenseFinder systems** (F1 0.060): 58% of their genes have
  no context-supported edge against 36% of all genes; their neighbourhoods vary between genomes.
- **Single input is viable.** Context-gated families match the MMseqs2 base, and plain
  similarity families from gene-level edges at high resolution (V0, a predeclared control
  outside the selection pool) exceed it on the summed excess (+0.038 on even, +0.033 on odd)
  with lower F1 on every caller. Round 1's embedding-only arms (unique-sequence mutual 20-NN,
  resolution 100-3000) were at -0.029 to -0.129.
- **Granularity is the trade-off.** TXSScan excess tracks keeping large families whole (base
  mean token size 1,869 for its members); DefenseFinder localization rewards small tokens. This
  revises round 1's reading of the TXSScan loss (fragmentation, not composition-driven merging).
- The dense identity holds numerically (2.2e-16); at windows of 3 and 5 the dense sum falls
  under the null tail, so only best-match context reaches 1% FDR beyond window 1.
Next hypothesis: one hierarchy over genes in place of one flat partition, with evidence read as
the size of the smallest shared cluster, for both the engine and locus search (workstream
`hierarchy`).

## 2026-11-12 09:43 UTC — Ribosomal-locus acceptance check (workstream `ribosomal`; `benchmarks/redesign/ribosomal/REPORT.txt`)

Jacob asked whether the redesign still resolves the ribosomal protein locus; it had not been
checked since the redesign began. Gold: per genome, runs of >= 16 genes with a `Ribosomal_*`
Pfam domain, up to 2 unmarked genes allowed inside (1,686 runs, 627 genera, 8 phyla; the
gap-free July definition yields runs in 10 genomes on the current gene set). Recovered = one
locus covers >= 50% of the run; resolved = one locus covers >= 80% with precision >= 0.8.
Controls: 20 length-matched windows per run on the same contig.

| Arm (window 3 unless noted) | Recovered | Resolved, default reading | Resolved, level 80 | Resolved, unit reading | Parvarchaeaceae |
|---|---|---|---|---|---|
| Base MMseqs2 e3c50 (engine v1.5) | 1,686 | 711 (precision 0.743) | 1,317 (0.923) | 149 | 114 / 114 |
| Random windows | 51.6% | 6.7% | | 0.02% | |
| Base, window 5 | 1,686 | 501 | 1,197 | 1,355 | 114 / 114 |
| Strict MMseqs2 id30c80 | 1,686 | 743 | 929 | 0 | 114 / 114 |
| Hybrid ESM++ merge, r0.05 | 1,686 | 770 | 1,266 | 1,565 (0.96) | 114 / 114 |
| Context-gated embedding families (dense w1) | 1,673 | 816 | 1,246 | 1,035 | 114 / 114 |
| Plain-similarity embedding families, r19000 (the +0.038 arm) | 1,155 | 466 | 0 | 98 | 6 / 114 |
| Plain-similarity embedding families, r11000 | 1,580 | 724 | 0 | 196 | 53 / 114 |

- The base engine finds the locus in every genome that has a run, as one locus in 1,640 of
  1,686. At the default reading the locus extends a median 3 genes past each end into conserved
  neighbours (the commonest is a SecY-domain family present in 3,280 genomes), which the
  precision rule counts as error; 958 of the 975 unresolved runs fail on precision alone.
- The run's genes fall into two modules (329 and 85) in every genome; top-module purity 0.849
  (random windows 0.149).
- Hybrid families (ESM++ merges of the base) make the unit reading work: 1,565 resolved against
  149 for the base, +248 [218, 278] over the base's best fixed reading. Each ribosomal family
  becomes one token.
- **The plain-similarity embedding families that exceeded the base on summed excess lose the
  locus** (1,155 of 1,686 recovered, 6 of 114 Parvarchaeaceae): fine tokens split universal
  families. That arm is disqualified as a default; the acceptance check is now part of
  `run_recurrence_dpann_arm.sbatch` and a guard for the hierarchy workstream.
- Larger windows lose 184-258 resolved runs at the default reading.

## 2026-11-12 09:53 UTC — Gene-call repair of the family tokens (workstream `repair`; `benchmarks/redesign/repair/REPORT.txt`)

Criteria fixed 07:18 UTC before any search or engine job: adopt a repair if, on even genomes,
the summed DefenseFinder + TXSScan excess does not fall (lower bound above -0.005) and F1 on
gold instances containing a repaired gene rises by >= 0.03 with an interval excluding 0.
Genome-wide MMseqs2 search (1,093,510 unique queries against all 2,547,528; 23 min in 4 shards).

| Arm | Genes changed | Tokens | Summed excess vs base | Touched instances | F1 on touched | Excess on touched | Adopted |
|---|---|---|---|---|---|---|---|
| rescue1 (singletons join by >= 80% coverage of the shorter protein) | 54,734 | 375,737 | +0.002 [-0.002, +0.007] | 52 | +0.301 [+0.202, +0.410] | +0.164 [+0.092, +0.249] | yes |
| **rescue3** (tokens with fewer than 3 members) | 83,466 | 363,307 | +0.004 [-0.003, +0.010] | 101 | +0.260 [+0.181, +0.340] | +0.166 [+0.112, +0.223] | **yes, preferred** |
| split (adjacent pieces take the intact homolog's token) | 44,287 | 429,642 | -0.002 [-0.005, +0.001] | 105 | -0.308 [-0.391, -0.229] | -0.253 | no |
| rescue1 + split | 98,244 | 375,631 | +0.000 [-0.005, +0.006] | 156 | -0.101 [-0.187, -0.019] | -0.112 | no |
| rescue3 + split | 126,565 | 363,234 | +0.002 [-0.005, +0.009] | 202 | -0.017 [-0.090, +0.061] | -0.043 | no |

- **Fragment rescue is adopted** (rescue3): DefenseFinder F1 over all 1,261 even-genome instances
  +0.020 [+0.013, +0.028]; genes covered by loci 2,256,894 -> 2,302,705; engine FDR 0.00425. Of
  20 random rescued genes, 17 are fragments joining their family, 2 full-length matches, 1
  domain-level glue (a 1,163-aa protein joining a 232-aa family at 17% query coverage).
  New DPANN default family table:
  `engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet`.
- **Split unification fails for a mechanical reason.** The engine counts pairs of different
  tokens only (`ta != tb`), so two pieces sharing a token lose the adjacency that located them:
  168 of 220 touched instances contain both pieces (F1 -0.326). 106 gold instances are
  two-member instances made of exactly one split pair (103 CRISPRCasTyper): in those, the "system
  of two members" is one gene called in two pieces. A fair test needs a merged gene universe
  (`merged_universe_design.txt`; not built).
- Ribosomal acceptance check: every arm recovers 1,686 / 1,686 runs and 114 / 114
  Parvarchaeaceae; top-module purity 0.849 (base), 0.974 (rescue1), 0.896 (rescue3).

## 2026-11-12 10:27 UTC — Hierarchy of families (workstream `hierarchy`; `benchmarks/redesign/hierarchy/REPORT.txt`)

Hypothesis: one multi-resolution hierarchy from one input, evidence = -log(size of the smallest
shared cluster / N), serves the engine and search. Predeclared 09:26 UTC; ribosomal guard added
09:44 UTC (admissible = >= 1,650 of 1,686 runs and >= 110 of 114 Parvarchaeaceae recovered).

- **T1, plain-similarity embedding families confirmed and inadmissible.** Three new Leiden seeds
  at resolution 19000: summed excess +0.039 / +0.038 / +0.038 over the base on even genomes, F1
  below the base on all three callers, and 1,122-1,165 ribosomal runs with 5-16 of 114
  Parvarchaeaceae. The step from round 1 (+0.067) splits evenly between gene-level capped edges
  (+0.052 [+0.033, +0.070] at about 155k tokens) and resolution (+0.049 [+0.027, +0.070]); no
  graph-by-resolution cell is admissible.
- **T2, multi-resolution recurrence: not met.** `recurrence_ml.py` (one token per level, smallest
  Poisson P per gene-pair instance with a multiplicity correction, joint null) reproduces v1.5
  exactly with one level. Embedding hierarchies are ruled by their coarsest level (63-77% of
  significant instances), and loci read from the union of significance only grow: admissible
  sets sit 0.074-0.114 below their finest level alone. Sequence hierarchy (id30c80 + e3c50):
  +0.004 [-0.002, +0.010]. Mixed (two MMseqs2 levels + three embedding levels): +0.012
  [+0.002, +0.022] over the base, admissible (1,686 / 114), DefenseFinder F1 0.849; below the
  +0.02 bar.
- **T3, hierarchy evidence in search with no per-query search: not met.** Embedding levels AP
  0.513, MMseqs2 levels 0.450, both 0.575 (M0 0.370, M2 0.576, S2 0.620, S3 0.698) at 0.16-0.27
  CPU-s per query against 8.7 for M2; 3.3-9.1x M0's unlabelled hits (bound 1.5x), 92-95% of
  them carrying the query type's profile domains; unsupported unlabelled hits per query 0.231
  for both sources (M2 0.002, S3 0.029). The finest levels carry the ranking (+0.089 AP for 13
  against 8 levels).
- **T4:** the saved Leiden levels are far from nested (32-54% of genes), and nesting by
  intersection changes nothing material.
Reading: with ESM++ small, one flat MMseqs2 partition (now with fragment rescue) remains the
best family source for the engine; neither the product kernel nor the hierarchy unifies the
inputs. Remaining route to a single input: a learned head that distils alignment evidence into
one vector (workstream `head`).

## 2026-11-12 11:55 UTC — Error rate of fragment rescue (workstream `repair`; `benchmarks/redesign/repair/REPORT_rescue_error.txt`)

Prompted by the Sharur session's ground-truth test on its own families (9.5% wrong-family joins,
22.9% false joins of singleton pieces). Same design on ELSA's rule and families: 6,000 complete
proteins from families of >= 3 members and 2,000 complete singletons, cut to an N- or C-terminal
piece (15-95% kept), intact original removed from the targets. Decision rule appended to
`predeclared.json` at 11:43 UTC before computing: default = strictest variant keeping >= 80% of
rescue3's touched-instance F1 gain with summed-excess lower bound above -0.005, wrong-family
below 3% and false joins below 8%.

| Rule | True family | Wrong family | Unplaced | False join (singletons) | Genes moved | Gain retained | Summed excess (even) |
|---|---|---|---|---|---|---|---|
| rescue3 (adopted earlier) | 89.3% | 5.0% | 5.7% | 13.4% | 83,466 | 100% (+0.260 F1) | +0.004 [-0.003, +0.010] |
| a: rescued gene is the shorter protein | 89.0% | 4.5% | 6.5% | 9.0% | 71,410 | 84% | +0.003 [-0.002, +0.009] |
| b: a + contig end or partial flag | as a | as a | as a | as a | 44,490 | 73% | +0.003 [-0.001, +0.008] |
| c: a + alignment reaches the target's terminus on the intact side | 85.8% | 3.6% | 10.6% | 4.0% | 42,087 | 70% | +0.003 [-0.002, +0.007] |
| d: b + c | as c | as c | as c | as c | 38,881 | 69% | +0.003 [-0.002, +0.007] |

- **No variant meets both error bounds**; by the rule nothing is selected. Variant a keeps the
  gain and misses both bounds; variant c meets the false-join bound and misses wrong-family by
  0.6 points.
- False joins grow with fragment length (rescue3: 4.7% under 75 aa, 26.2% at >= 300 aa; c: 2.5%
  and 4.0%). Of the wrong joins, 103 of 301 (rescue3), 111 of 271 (a) and 111 of 218 (c) land in
  a token whose member aligns full-length to the intact original; excluding those, wrong-family
  is 3.3%, 2.7% and 1.8%.
- Every variant passes the ribosomal check (1,686 / 1,686; 114 / 114). DefenseFinder F1 over all
  even-genome instances rises +0.012 to +0.020 under every arm.
- The default family table is a decision for Jacob (rescue3, a, c, or no rescue).

## 2026-11-12 12:43 UTC — Decision (Jacob): the current fragment-rescue rule stays

Jacob chose to keep rescue3 (a gene in a family of fewer than 3 members joins its best hit's
family when the alignment covers >= 80% of the shorter protein) as the DPANN default family
table, knowing its cut-test error (5.0% wrong family, 13.4% false joins of singleton pieces).
The anchored variant (`rescue3c`) stays available as the conservative table.

## 2026-11-12 13:03 UTC — CORRECTION: whitened embedding ranks beat MMseqs2 ranks in locus search (workstream `head`; `benchmarks/redesign/head/REPORT.txt`)

**Error in locus-search rounds 1 and 2 (the coordinator's brief):** the embedding ranks there
(M1, M2, S3) were computed on the uncentred L2 vectors (`l2_f32__row.npy`). On DPANN, raw
ESM++ small vectors are strongly anisotropic; homolog retrieval AP@2000 is 0.336 for raw L2
against 0.722 after the zca10 transform (mean removal and soft whitening, unsupervised), in line
with the July transform ablation. The statements "sequence search supplies graded evidence
better than ESM++ small" and "embedding ranks alone 0.442" hold for unwhitened vectors only.

Even-genome DefenseFinder queries (1,247; 200-replicate null; configurations tuned on odd):

| Method | Inputs | AP | Strict AP | Recall E<=0.01 | Unlabelled / query | Unsupported / query |
|---|---|---|---|---|---|---|
| M0 tokens | 1 | 0.370 | 0.256 | 0.094 | 0.72 | 0.000 |
| M1 raw L2 embedding ranks | 1 | 0.442 | 0.257 | 0.093 | 2.59 | 0.137 |
| S1 MMseqs2 ranks | 1 | 0.602 | 0.389 | 0.170 | 4.47 | 0.029 |
| S3 tokens + MMseqs2 + raw L2 | 3 | 0.698 | 0.454 | 0.182 | 4.51 | 0.029 |
| **M1z zca10 embedding ranks** | 1 | **0.792** | 0.510 | 0.183 | 4.51 | 0.121 |
| M1' learned-head ranks (linear 960 -> 256) | 1 | 0.809 | 0.516 | 0.192 | 4.49 | 0.065 |
| M2' tokens + head | 2 | 0.821 | 0.521 | 0.184 | 3.63 | 0.020 |
| S3' tokens + MMseqs2 + head | 3 | 0.826 | 0.521 | | | 0.038 |

- Whitened embedding ranks alone exceed MMseqs2 ranks alone by 0.19 AP and the three-input S3 by
  0.09. TXSScan: M1' 0.931, M1z 0.868, S3 0.747. CRISPRCasTyper: 0.154, 0.135, 0.133 (with more
  wrong-subtype hits). The M1z arms were added by amendment during the run, so the comparison
  needs a predeclared confirmation (workstream `search3`).
- The trained head adds little: +0.017 [+0.013, +0.021] AP over zca10 and half the unsupported
  rate; retrieval AP@2000 +0.075 over zca10 (bar 0.10). A linear map beat every MLP.
- Predeclared outcomes of the head test: H1 not met on the rate condition (AP 0.809 passes; 0.065
  unsupported unlabelled hits per query against a bound of 0.05 taken from a different null);
  H2 not met (admissible head-family arm -0.070 [-0.087, -0.054] against the MMseqs2 base; the
  only arm above the base loses the ribosomal locus); H3 not met.
- Discovery is unchanged: MMseqs2 families with fragment rescue remain the engine default.
- Affected earlier statements: locus-search rounds 1-2 conclusions about the embedding (S2 - M2,
  "the embedding reorders the deeper list", token-invisible recall, the 20,000 against 2,000 list
  depth) and the phase-0 "embedding-reachable" counts, all computed on raw vectors.
- The Mac's data volume reached 100% during this run (10:41 UTC) and was freed by another
  session's cleanup.

## 2026-11-12 13:23 UTC — Direction from Jacob before the overnight runs

- Omnitrophota is not next: the method is not settled. No second collection until it is.
- The split design (MMseqs2 families for discovery, whitened embeddings for search) is a
  compromise he does not want. Target: one method on one input, with the embeddings doing work
  alignment cannot, serving discovery and search alike.
- Overnight tests that bear on this: `softrec` (token-free recurrence from whitened neighbour
  lists; the direct test of one statistic for both tasks), `grammar` (adjacency critic scored on
  held-out orders), `search3` (predeclared confirmation of whitened-embedding search),
  `baseline` (the old chaining pipeline with whitening on the DPANN gold loci), `boundary`
  (locus reading; split genes as one position).

## 2026-11-12 13:56 UTC — Locus reading rules; every DPANN F1 so far is a best-of-11-readings figure (workstream `boundary`; `benchmarks/redesign/boundary/REPORT.txt`)

**Scoring fact that qualifies all DPANN numbers above.** `score_dpann_loci.py` takes, per gold
instance, the best locus in the whole `loci_all_genomes.parquet` table, and the engine writes 11
readings into it (conserved at several levels, unit readings). The F1 values quoted for engine
arms (for example DefenseFinder 0.806-0.814, TXSScan 0.753-0.758, CRISPRCasTyper 0.586-0.589)
are best-of-11. The default single reading (conserved at level 2) on even genomes, rescue3 arm:
DefenseFinder 0.688, TXSScan 0.409, CRISPRCasTyper 0.426; summed DefenseFinder + TXSScan excess
0.337. Arm-against-arm comparisons were all scored the same way, so their signs stand; absolute
values and any comparison with a method that outputs one table need the single-reading figure.

**Reading rules (tuned on odd genomes, 172 configurations; even genomes against the default
reading R1; predeclared conditions: ribosomal resolved >= 1,100 with recovery kept, summed excess
+0.02, no caller losing more than 0.01 F1).**

| Rule | F1 change DF / TXSS / CCT | Summed excess change | Ribosomal resolved / recovered | Parvarchaeaceae resolved | Verdict |
|---|---|---|---|---|---|
| A trim ends | +.017 / +.046 / -.023 | +.084 [+.074, +.094] | 1,165 / 1,682 | 63 | fails F1 |
| B support segmentation | +.058 / +.084 / +.030 | +.066 [+.051, +.081] | 929 / 1,680 | 106 | fails ribosomal |
| C module cuts (Louvain resolution 0.3) | +.033 / +.097 / +.041 | +.045 [+.034, +.056] | 1,435 / 1,686 | 103 | passes |
| D level-2 locus + level-200 core | +.001 / +.226 / +.018 | +.171 [+.150, +.191] | 1,377 / 1,686 | 13 | passes; adopted by the tie-break |
| G C then support segmentation | +.080 / +.154 / +.057 | +.101 [+.084, +.118] | 1,448 / 1,680 | 107 | passes |
| H C then end trimming | +.050 / +.136 / +.032 | +.113 [+.099, +.126] | 1,554 / 1,685 | 114 | passes |

- Against the full 11-reading table (R2) no one- or two-reading output keeps F1 (0.06-0.32
  lower); on summed excess D exceeds R2 by +0.058 [+0.038, +0.077] and H equals it with one
  reading (-0.000 [-0.016, +0.016]).
- G and H lose coverage (H: genes in loci 2,563,690 -> 2,308,785; CRISPRCasTyper covered50
  -0.031); C and D keep every locus.
- Module labels mark system edges (AUROC of "boundary inside a system" against overshoot: 0.83 /
  0.75 / 0.74 by caller); support level alone points the wrong way for CRISPR-Cas (0.377). At
  Louvain resolution 0.3, 1,547 of 1,686 ribosomal runs carry one module; at the engine's
  resolution 10 about 1,570 carry two.
- The excess gains have the same size on the base, rescue3 and hybrid arms (the rule does the
  work); the ribosomal result of module-based rules depends on the families.
- Follow-up running: level-2 locus + H core as a nested output; a per-reading inventory.

**Split genes merged into one position: not adopted.** 38,651 pairs merged; 106 gold instances
leave the >= 2-member set; F1 on instances containing a split pair +0.018 [-0.049, +0.110]
(n = 29), unaffected instances -0.001 [-0.002, +0.001].

## 2026-11-12 14:03 UTC — Nested reading adopted: level-2 neighbourhood + module-cut, end-trimmed core (`boundary/REPORT_followup.txt`)

Predeclared 13:56 UTC, no tuning (Louvain resolution 0.3, seed 17, trimming factor 0.5).
Rescue3 arm, even genomes, against the default single reading R1:

| Caller | F1 nested / R1 | F1 change | Excess change | Covered50 nested / R1 |
|---|---|---|---|---|
| DefenseFinder | 0.744 / 0.688 | +0.056 [+0.048, +0.065] | +0.019 [+0.012, +0.026] | 0.924 / 0.924 |
| TXSScan | 0.548 / 0.409 | +0.139 [+0.125, +0.154] | +0.071 [+0.060, +0.082] | 0.990 / 0.990 |
| CRISPRCasTyper | 0.479 / 0.426 | +0.053 [+0.048, +0.059] | +0.008 [+0.004, +0.012] | 0.890 / 0.890 |

Summed DefenseFinder + TXSScan excess +0.090 [+0.077, +0.102]. Ribosomal: 1,686 recovered,
1,512 resolved (R1 704; level-200 core 1,377; trimmed module reading alone 1,554), median
precision 0.96, Parvarchaeaceae 113 of 114 resolved. Genes in loci unchanged (2,563,690). It
replaces the level-200 core as the adopted reading rule. Against the full 11-reading table it
is 0.08-0.21 F1 lower per caller and -0.023 [-0.039, -0.007] on summed excess.

Reading inventory (rescue3, even genomes): the unit reading at Jaccard 0.5 alone gives F1 0.809 /
0.721 / 0.568 (full table 0.826 / 0.757 / 0.597), is the best reading for 81% / 77% / 74% of
instances and the only best one for 12% / 39% / 33%, and resolves 149 ribosomal runs. The
conserved level-2 reading is best for 50% / 11% / 34%. A three-reading output (neighbourhood,
core, unit) is untested. Integration into the engine waits for the outcome of `softrec`.

## 2026-11-12 14:52 UTC — The old chaining pipeline on DPANN, scored like the engine (workstream `baseline`; `benchmarks/redesign/baseline/REPORT.txt`)

Legacy pipeline (kNN k 50 on ESM++ small, chaining with gap 2, minimum 2 anchors) on all 3,564
genomes, nothing capped: raw cosine >= 0.85 gives 160.2 M anchors and 7,626,476 blocks (81.9%
with 2-3 anchors; 33.5 min on 64 CPUs for kNN and chaining); soft-ZCA alpha 0.1 with cosine >= 0.30
gives 155.4 M anchors and 10,750,829 blocks (80.8%). Both thresholds keep 97-98% of retrieved
neighbours.

**Like for like, even genomes, summed DefenseFinder + TXSScan excess:**

| Output | Kind | Summed excess |
|---|---|---|
| Engine rescue3, default reading (conserved level 2) | single | 0.338 [0.313, 0.361] |
| Engine rescue3, trimmed module reading (rule H) | single | 0.451 [0.427, 0.473] |
| Legacy, union of blocks per contig, raw | single | 0.181 [0.160, 0.201] |
| Legacy, union of blocks per contig, soft-ZCA | single | 0.183 [0.163, 0.203] |
| Engine rescue3, full table | best of 11 | reference |
| Legacy, all block intervals, raw / soft-ZCA | best of many | -0.082 [-0.098, -0.066] / -0.086 [-0.100, -0.072] against the engine full table |

- Legacy all-interval tables have the highest absolute DefenseFinder F1 of anything scored (0.830
  raw, 0.873 soft-ZCA; engine full table 0.826) with a higher random-placement control (0.60-0.64
  against 0.55): a median 93-105 legacy intervals cover each ribosomal run against 7-8 engine loci.
- Ribosomal check, single readings: legacy union resolves 406-444 of 1,686 runs (best union locus
  a median 41-44 genes for 24-gene runs); engine default 704-711; rule H 1,554. Best-of-many:
  legacy soft-ZCA 1,385, engine full table 1,370 (+15 [-27, +56]); legacy raw 1,009.
- Boundary precision, single readings (DefenseFinder / TXSScan / CRISPRCasTyper): legacy union
  0.375 / 0.12 / 0.13; engine default 0.67 / 0.27 / 0.32; rule H 0.75 / 0.40 / 0.375.
- The Spacedust calibration rule fails on DPANN: permuted-order blocks have higher mean anchor
  cosine than observed ones at every size (chance size-2 blocks fall between genomes of one
  species 29.7% of the time against 1.0% for observed), so "calibrated" reduces to a minimum of
  4-5 anchors and loses two-gene systems (DefenseFinder F1 0.15-0.24).

**CORRECTION to the Spacedust comparison quoted on 12 November.** The engine's source207 figures
("conserved loci, gaps 0-2": 0.745-0.801) are a best of 42 readings (3 gap rules x 14 support
levels), and the legacy figures (0.652-0.762) are a best over all block intervals. As single
readings: engine conserved level 2 F1 0.570-0.651 with excess +0.164 to +0.222; union of
calibrated legacy blocks F1 0.331-0.491 with excess +0.070 to +0.125; the engine leads by +0.065
to +0.152 excess, all 21 paired intervals above 0.

## 2026-11-12 14:58 UTC — Whitened-embedding search confirmed for ranking; a second input still buys recall at matched false-positive rate (workstream `search3`; `benchmarks/redesign/locus_search/REPORT3.txt`)

Predeclared 13:10 UTC (`results/criterion3.json`). DefenseFinder, 5-fold genome cross-validation
(2,447 queries); every fold chose the same configurations.

- **Confirmed:** zca10 ranks alone (M1z) AP 0.796 against MMseqs2 ranks alone (S1) 0.610 and the
  three-input S3 0.706; M1z - S1 = +0.186 [+0.179, +0.194], M1z - S3 = +0.090 [+0.085, +0.095];
  strict AP +0.118 and +0.053. TXSScan: +0.390 and +0.120. CRISPRCasTyper: +0.016 and +0.002
  [-0.000, +0.004], with wrong-subtype hits per query rising to 17.4 (S3 6.1).
- **Controls:** shuffled lists 0.062; a length-and-composition ranker 0.194 (tokens 0.369).
  Whitening alone 0.777, centring alone 0.483, per-genome centring 0.513, raw L2 0.445: whitening
  does about 95% of the work. Fitting the transform on odd genomes only changes AP by +0.0002.
  List depth: top 2,000 loses 0.098 AP, top 200 loses 0.455; exact top 10,000 is within 0.008.
- **At a matched false-positive rate the single input trails.** At E <= 0.01 on the calibrated
  null, unsupported unlabelled hits per query: M1z 0.070, S1 0.014, S3 0.017. At a matched 0.03
  unsupported hits per query, recall M1z 0.122, S1 0.175, S3 0.186 (M1z - S1 -0.053
  [-0.060, -0.046]); at matched wrong-type rates of 0.25 / 0.5 / 1 per query M1z - S1 is -0.044 /
  -0.040 / -0.030. Adding to M1z at the matched rate: tokens +0.032 recall, MMseqs2 ranks +0.051,
  the learned head +0.054. M1z's unsupported hits (87, from 41 queries) mostly carry a
  DefenseFinder profile of another type (79%) and rarely have sequence homology to the query
  (17%).
- **Redone on whitened vectors:** 51.5% of same-type DefenseFinder instance pairs are
  token-invisible and embedding-reachable (raw: 20.6%; different-type control 14.8%). Recall on
  token-invisible targets at E <= 0.01 / E <= 1: M1z 0.075 / 0.276, S1 0.046 / 0.131, raw M1
  0.017 / 0.073. The iterative profile with zca10 centroids still drifts (wrong-type hits 3.0-3.7x
  by round 3, no AP gain).
- **One-input command:** exact brute force over the whitened vectors in memory (13.8 GB), median
  1.33 s per query on 24 CPUs, 0.54 s with the null replicates threaded (round-2 command 4.75 s).
  HNSW never came within 0.01 AP of exact top-20,000.
- The learned linear head: +0.017 [+0.013, +0.021] AP over zca10 on held-out queries and +0.054
  recall at the matched unsupported rate.

## 2026-11-12 16:26 UTC — Token-free recurrence from whitened neighbour lists (workstream `softrec`; `benchmarks/redesign/softrec/REPORT.txt`)

Each gene's ranked zca10 neighbour list (genes outside its genus) replaces its family token in
the recurrence statistic; the kernel reproduces the token engine exactly when the list is the
family (L and G equal for all 9,350,064 instances; all 2,939,492 loci equal). Predeclared 13:27
UTC; variant chosen on odd genomes among admissible ones: zca10, top 1,000, count rule L >= 15
(instance FDR 0.0046).

| Comparison (even genomes) | Summed DF+TXSS excess | DefenseFinder F1 | TXSScan F1 | CRISPRCasTyper F1 | Ribosomal recovered / resolved / Parv. |
|---|---|---|---|---|---|
| Full table: softrec against base | +0.061 [+0.041, +0.080] | 0.843 vs 0.806 | 0.699 vs 0.758 | 0.499 vs 0.589 | 1,681 / 984 / 114 |
| Default single reading: softrec against base | +0.054 [+0.032, +0.075] | 0.711 vs 0.673 | 0.448 vs 0.412 | 0.412 vs 0.422 | (base 1,686 / 711 / 114) |
| Softrec end-trimmed reading against token rule H | -0.028 [-0.049, -0.007] | 0.731 vs 0.723 | 0.460 vs 0.558 | 0.408 vs 0.451 | 1,679 / 1,104 / 114 |

- **Predeclared outcome: neither matches nor beats**, because full-table TXSScan F1 is 0.059 below
  the base (bars 0.02 and 0.01), although summed excess is +0.061.
- **First embedding-only discovery arm that is above the base and keeps the ribosomal locus.** At
  the default single reading it beats the token engine on summed excess and on DefenseFinder and
  TXSScan F1.
- The machinery alone (family lists; G and E counted over other genera) gives +0.011 to +0.014 at
  unchanged TXSScan F1; embedding lists add about +0.05 through DefenseFinder localization.
- Where it loses: TXSScan in the locus reading (the unit reading over-merges because a bounded
  list fills with close relatives; no admissible softrec single reading reaches the token engine's
  module-based rule H, which needs token-pair modules); CRISPR-Cas in the statistic (at depth
  1,000 the null forces L >= 15 and variable Cas pairs fall under it: member genes touching
  support 0.616 vs 0.765).
- Depth 50 gains most (+0.064 to +0.068) and loses Parvarchaeaceae runs (97-101 of 114); the
  Poisson statistic fails on deep lists. The learned head trails zca10 at every admissible depth.
- Cost: 29 min for six variants at depth 1,000 on 24 CPUs after a 2 h 14 min kNN (top 2,000).
Follow-up `softrec2`: token-free modules for the reading rule, a two-tier statistic, and the
lineage-aware expectation, judged on single outputs only.

## 2026-11-12 18:55 UTC — Census of unannotated blocks and co-fold screen 3 (workstream `census`; `benchmarks/redesign/cofold/REPORT_census.txt`)

Engine rescue3 run; Sharur DPANN database republished 12 November (families, remote-homology
tier for 33,724 Pfam-dark families, CDD).

**Census: 22,531 block types** (2-8 adjacent unannotated genes linked by significant family
pairs, >= 3 genomes; 253,328 instances, 263,825 genes), by the most-recognised chain:

| Category | Types | >= 5 genomes and >= 2 orders | Genes |
|---|---|---|---|
| A chain's family has annotated members elsewhere | 12,926 | 1,548 | 171,108 |
| A chain resembles a characterised family (profile search) | 2,286 | 278 | 31,904 |
| A chain has a CDD-only hit | 403 | 52 | 4,484 |
| Unrecognised chains, one or more outside the remote-homology tier | 2,901 | 217 | 37,645 |
| Unrecognised, unknown-function families only | 522 | 81 | 8,270 |
| Unrecognised, no hit in any chain | 3,493 | 557 | 61,724 |

Of 2,563,690 genes in engine loci, 68,599 (2.68%) sit in a block with no recognised chain
(23,494, 0.92%, at >= 5 genomes and >= 2 orders). The automated tier (family-only alignments) is
less sensitive than the enriched hand triage: of 50 searched families that triage classed
characterised it calls 33 resembling, 2 CDD-only, 2 unknown-only and 13 no hit, so "no hit" in the
tier is weaker than "unrecognised".

**Screen 3** (criteria frozen 17:00 UTC before any fold): of 4,015 all-unrecognised types, 206
pass the foldability filters; 85 were folded in screens 1-2; 94 new non-redundant blocks were
submitted (435 items; 81 blocks complete, 13 blocks with complexes above 768 aa still queued at
the API). Provisional class rates: engine-called 12 / 165 folds with a passing interface (7.3%),
partner swaps 1 / 82, uncalled windows 1 / 81, curated positives 5 / 35 (ipTM distributions:
called against uncalled p 3e-6; binary rate p 0.066).
- Four blocks pass in both replicates. Three are known units: two variants of the showcase A-B
  pair (a B-variant family; a minority C family; the caspase-like gene again directly
  downstream in 12 of 13 instances) and the PIN / antitoxin-like operon of screen 1 with its
  third gene. The fourth (tokens 205406-263312) passes every predeclared check, but its two
  replicates carry identical protein sequences found in MAGs of two phyla; the one independent
  sequence variant does not co-fold (ipTM 0.358). Refuted post hoc.
- **No new block is both unrecognised by every search and co-folding across independent
  sequences** (pending the 13 queued blocks). The A-B-C operon remains the single showcase.
- Design flaw: choosing replicates by taxonomic distance alone picked identical or near-identical
  sequences for 12 of 94 blocks; replicate selection needs a sequence-identity ceiling.
- Biohub items: 462 (ledger `work/s3/biohub_ledger.json`); items expire 17:00 UTC 13 November.

## 2026-11-12 20:21 UTC — Neighbourhood-grammar critic (workstream `grammar`; `benchmarks/redesign/grammar/REPORT.txt`)

Critic s(a, b) = u(x_a) . v(x_b) on genome-centred zca10 ESM++ small vectors, trained to tell
adjacent gene pairs from non-adjacent pairs of the same genome; 5-fold cross-validation grouped
by GTDB order, plus a train-on-one-phylum split. Baseline: token-pair PMI from the same training
orders. 16.4 h of 64-CPU jobs; no GPU.

| Held-out orders, AUROC of curated-system adjacencies against other adjacencies | Critic | Training-order PMI |
|---|---|---|
| DefenseFinder | 0.824 | 0.785 |
| TXSScan | 0.852 | 0.811 |
| CRISPRCasTyper | 0.685 | 0.643 |
| Token pairs never co-located in the training orders | 0.768 | 0.544 |
| Pairs touching a singleton family | 0.786 | 0.500 |
| Pairs seen in 1-4 training genomes | 0.801 | 0.802 |

- **Predeclared: G1 not met** (overall +0.039 [+0.029, +0.048] against a bar of +0.05; the
  never-seen part +0.224 [+0.180, +0.267] against +0.10 is met; both were required). **G2 not
  met** (co-folding AUROC 0.692 [0.604, 0.770] against 0.748 for the engine's significance; the
  four uncalled pairs that co-folded rank 53, 153, 198 and 211 of 297). **G3 met** (the critic
  trained on shuffled order scores 0.519).
- The signal is in the pair: the interaction part alone gives 0.814, the per-gene part 0.290.
  With raw cosine held fixed the never-seen gain holds (0.743 vs 0.547) and the overall gain
  falls to +0.023; tandem homologs split across families account for about 40% of the overall
  gain. Where a pair has been seen, counts mark system boundaries better (PMI 0.870, engine
  0.845, critic 0.830).
- Phylum split (train on Nanobdellota, score the other eight phyla): +0.050 [+0.034, +0.064]
  overall, +0.239 on never-seen pairs.
- Loci read from the critic: summed excess +0.586 against +0.444 for the engine default with
  lower F1 on every caller (DefenseFinder 0.667 vs 0.833) and a failed ribosomal guard (1,360 of
  1,686 runs; 31 of 114 Parvarchaeaceae).
- Paralogs aside, 28 of the top 30 uncalled high-scoring pairs are one recurring pair type over
  28 different family pairs: a protein of about 50 aa beside a longer one (180-1,070 aa); observed
  domains on embedding neighbours only (pilin-like signal-peptide families on the short one;
  lytic transglycosylase, M23 peptidase, PKD on the long one). UNVERIFIED as any system. Six
  candidate pairs are listed for a co-fold screen; none folded.
Reading: the critic complements family counts exactly where counts are blind (pairs unseen in
other lineages, singleton families) and does not replace them.

## 2026-11-12 20:34 UTC — Token-free recurrence, round 2 (workstream `softrec2`; `benchmarks/redesign/softrec/REPORT2.txt`)

Single outputs only; even genomes; references reproduced exactly (token default reading R1:
F1 0.688 / 0.409 / 0.426, summed excess 0.3375; token nested reading DH: 0.744 / 0.548 / 0.479,
summed 0.4270, ribosomal 1,686 recovered and 1,512 resolved). Predeclared in `predeclared2.json`.

Selected on odd genomes: rank-weighted lists (depth 1,000; rule W >= 1 and L >= 2; FDR 0.0028),
families built from the statistic's own supporting pairs (Leiden), modules at Louvain 0.3, nested
reading.

| Output (even genomes) | Summed excess vs token DH | DefenseFinder F1 | TXSScan F1 | CRISPRCasTyper F1 | Ribosomal resolved (recovered) |
|---|---|---|---|---|---|
| Token engine, nested reading (reference) | 0 | 0.744 | 0.548 | 0.479 | 1,512 (1,686) |
| Embedding only, nested reading | +0.040 [+0.022, +0.059] | 0.802 | 0.500 | 0.442 | 1,263 (1,681); Parv. 114 recovered, 100 resolved |
| Embedding statistic + MMseqs2 families as module tokens | +0.077 [+0.058, +0.096] | | +0.001 vs DH | -0.039 vs DH | 1,499 |

- **Predeclared verdict: the embedding-only method does not match the token engine's nested
  reading.** Blocking conditions: TXSScan F1 (-0.048 [-0.061, -0.034]), CRISPRCasTyper F1 (-0.038
  [-0.048, -0.028]) and ribosomal resolved (1,263 against a bar of 1,400). Summed excess clears
  the "beats" bar and DefenseFinder F1 is +0.059.
- **The validated families are the limiting part of the reading.** They split by lineage: a
  ribosomal Pfam marker is spread over a median 423 families (MMseqs2: 5), the largest holding 18%
  of its genes (MMseqs2: 90%); Parvarchaeaceae ribosomal runs carry one module label in 2 of 114
  cases (MMseqs2: 114). With MMseqs2 families as module tokens on the same embedding landscape the
  TXSScan and ribosomal deficits vanish.
- **The CRISPR-Cas loss is in the statistic** (-0.038 with either family source). A two-tier
  statistic restores Cas member support (0.616 -> 0.728; token 0.774) at a cost of summed excess
  (-0.021 against token R1, where the count rule gives +0.050).
- Rank-weighted lists are best at the default reading (depth 200: +0.106 [+0.087, +0.125] over
  token R1) and keep the ribosomal locus at every depth.
- The lineage-aware expectation on the token engine is not confirmed on single outputs (DH
  -0.009 [-0.018, -0.000]); round 1's +0.014 was a best-of-11 effect.
- Cost on one 64-CPU node, vectors to loci: about 45 min at depth 200 (kNN 34 min), 103 min at
  depth 1,000; depths 200 and 500 keep the verdict.
Next: neighbour lists stratified by lineage (one best match per genus), which is what the
statistic counts, to stop lists and families from filling with close relatives (workstream
`softrec3`).

## 2026-11-12 23:35 UTC — Lineage-stratified neighbour lists (workstream `softrec3`; `benchmarks/redesign/softrec/REPORT3.txt`)

Hypothesis: one best match per genus (what the statistic counts) stops lists and derived
families from filling with close relatives and lowers chance support. Predeclared 20:52 UTC;
selection recorded before any even-genome discovery score (the agent was interrupted once by an
account limit; order of events unaffected).

- **Refuted on both parts; the selected configuration neither matches nor beats.** Even genomes
  against the token engine's nested reading: summed excess +0.130 [+0.099, +0.162], TXSScan F1
  +0.070, DefenseFinder F1 0.466 (-0.278), CRISPRCasTyper F1 0.272 (-0.207), ribosomal 1,648
  recovered and 1,276 resolved (Parvarchaeaceae 1 of 114).
- **A per-genus best match is mostly not homology at a pair-level floor.** zca10 cosines are
  asymmetric and the per-genome maximum has no valley: in a genome lacking the query's MMseqs2
  family the best match passes cosine 0.32 in 35% of cases, 0.45 in 14%, 0.60 in 4.2% (family
  present: 95%, 80%, 54%). Exact stratified lists reach 95% of the genera holding a family member
  (global top-1,000: 53%), and 57% of the genera they reach hold none.
- Chance support falls 2.6-2.9x with a per-genus cap, but count thresholds stay at L >= 8-30 and
  all 54 candidates fail the CRISPR-Cas bar at the default reading. Leiden families from
  stratified pairs agree better with MMseqs2 (AMI 0.83-0.86 against 0.66) and still cut the
  Parvarchaeaceae ribosomal runs; connected components keep the runs whole by merging markers
  (purity 0.36).
- Post hoc, closest to matching: rank-weighted r = 1000 with count-matched Leiden families:
  summed +0.057, F1 -0.032 / -0.012 / -0.083, 1,341 resolved.
- Search with the same structure: AP 0.562 against 0.792 for whitened top-20,000 ranks; recall at
  a matched wrong-type rate +0.057. One stratified structure does not serve both tasks.
- Exact per-genome best match for all sequences takes 8 min 43 s on one A5000 (HNSW top-1,000 on
  64 CPUs: 68 min); vectors to loci about 20 min of compute.
Round 2's configuration remains the best embedding-only discovery output. Reading: at long range
the whitened ESM++ small similarity does not separate homologs from chance well enough for a
per-lineage best match; the representation is the limit this lever ran into.

## 2026-11-12 23:39 UTC — Sharur DPANN database republished (note from the Sharur session)

Seal 78c68b6b…, 16:38 MST 12 November. CDD hits moved from `annotations` (source 'cdd') to table
`protein_resemblance`; a stored view `family_annotations_carried` carries family annotations to the
rescue clustering runs. Family tables unchanged. `cofold/s3_export_sharur.py` reads CDD from
`annotations`; point it at `protein_resemblance` before any rerun of the census export. The census
numbers already reported came from the earlier seal (f652ef47…).

## 2026-11-12 23:53 UTC — Constraint from Jacob: no taxonomy needed to run the pipeline

"Don't do things like per-genus matching; I don't want to make GTDB-Tk a necessary prerequisite
to run this pipeline." The per-genus list experiment is closed (and was refuted anyway). The
constraint reaches further than that experiment: **the engine's lineage unit is the GTDB genus**
(`recurrence.py`: `self.lineage` from `genomes.parquet` `genus`; one lineage per genome when
missing). The genus count L enters the significance rule (L >= 2), the support landscape and
reading thresholds, and the module edge weights; the token-free variant also excludes a gene's
own genus from its list. Phylum is reporting only. Taxonomy stays allowed for evaluation
(benchmarks, the ribosomal check's family count, held-out-order splits).
Workstream `lineagefree`: a lineage unit derived from the pipeline's own data (genome groups or
effective genome weights from shared-family or embedding best-match similarity), threshold set
by an internal criterion, predeclared to replace the genus if the nested reading's summed excess
stays within 0.01, no caller loses more than 0.01 F1, and the ribosomal check holds.

## 2026-11-13 01:05 UTC — Token-free recurrence, round 4 (workstream `softrec4`; `benchmarks/redesign/softrec/REPORT4.txt`)

Four levers on round 2's embedding-only landscape, same match/beat criterion against the token
engine's nested reading (even genomes; single outputs; predeclared 23:32 UTC, selection written
00:38 UTC before any even score). References reproduced exactly first.

- **Verdict: neither matches nor beats.** Predeclared configuration (orientation mode 2 +
  instance-graph module labels + trimming factor 0.8): summed excess +0.040 [+0.021, +0.059],
  DefenseFinder F1 0.791 (+0.047), TXSScan F1 0.518 (-0.030 [-0.044, -0.017]), CRISPRCasTyper F1
  0.446 (-0.033 [-0.043, -0.023]), ribosomal 1,682 recovered / 1,348 resolved (bar 1,400).
- **Levers alone against round 2's configuration** (summed excess; resolved, round 2 = 1,263):
  module labels from communities of the instance graph, no families: +0.009 [-0.000, +0.018],
  1,423. Orientation-aware support: -0.020 [-0.029, -0.010], 1,257. Critic extension: -0.014
  [-0.018, -0.010], 1,264 (added genes are gold members at the base rate, 0.018-0.024 against
  0.019). Reading parameters (trimming 0.8, Louvain 0.5): +0.015 [+0.009, +0.020], TXSScan F1
  +0.023, 1,332.
- **Instance-graph labels work once overlap edges are added.** Support edges alone give
  pair-type communities that cut every ribosomal run (0 of 1,686 keep one label) while scoring
  F1 0.831 / 0.661 / 0.491 on odd genomes; overlap edges (weight 0.3, Leiden resolution 1)
  restore the run. System edges are marked less often than with token modules (0.085 against
  0.195).
- **Orientation halves chance support and does not pay.** Null instances with L >= 2 at depth
  1,000: 7.80 M unoriented, 5.79 M with the strand flag, 3.72 M with the full arrangement. On the
  token engine (`recurrence.py --oriented`) the nested reading moves -0.001 [-0.010, +0.007].
- **The predeclared combination rule admitted orientation and it cost the result.** Post hoc
  (after the verdict), labels + reading parameters on the unoriented landscape: summed +0.065
  [+0.047, +0.084], DefenseFinder F1 +0.062, TXSScan -0.017 [-0.031, -0.003], CRISPR-Cas -0.037
  [-0.047, -0.027], ribosomal 1,681 recovered / 1,404 resolved. Every "matches" bar except
  CRISPR-Cas.
- **CRISPR-Cas is the one deficit no lever closed in four rounds:** F1 0.437-0.455 in every
  configuration that keeps the ribosomal run (token 0.479); Cas member genes touching support
  0.656-0.677 against 0.774. The embedding rule needs W >= 1 or L >= 8-15 where the token rule
  needs L >= 2; a system confined to few lineages may sit under that floor (untested; next).
- The lineage unit is now one pluggable genome-to-lineage array in `softrec4.py` and
  `readings4.py`; no round-4 step reads taxonomy beyond it.
Next: a diagnosis of what Cas pairs lack (list depth, homologs per Cas gene, lineage spread,
threshold floor), then a rerun with the data-derived lineage unit from `lineagefree`.
