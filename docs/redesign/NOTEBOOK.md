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
