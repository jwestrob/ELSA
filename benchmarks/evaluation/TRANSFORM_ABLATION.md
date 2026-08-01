# Embedding post-processing for ELSA: measured effect

Ablation of embedding post-processing against the anisotropy of mean-pooled PLM
embeddings. Ladder, corpora and protocol follow the plan in
`benchmarks/PCA_NORMALIZATION_PLAN.md`, whose own March 2026 conclusion is void —
see [Relationship to the March 2026 result](#relationship-to-the-march-2026-result).

**Result in one line:** post-processing pays where retrieval is hard and pays
nothing where it is already saturated, so the benefit is set by corpus
divergence rather than by the transform.

---

## The anisotropy

Measured on DPANN (3,395,472 ESM2 vectors, 320-d, L2-normalised):

| quantity | value |
|---|---|
| ‖mean of the unit vectors‖ | 0.9318 |
| isotropy (participation ratio) | 0.1187 |
| cosine between two unrelated proteins | median 0.935 |
| cosine between anchored pairs | median 0.996 |

87% of every embedding is one shared direction, and the discriminative signal
occupies the last 6% of the cosine range. This is why `similarity_threshold` is
inert: moving it 0.85 → 0.95 removes 0.9% of retained anchors.

ELSA's ‖μ‖ = 0.932 sits above the entire range tested in the text-embedding
literature (arXiv 2511.11041 spans 0.19–0.85), so the size of the effect here
was an extrapolation until measured.

---

## Tier 1 — anchor-level retrieval (DPANN, Pfam domain architecture)

Two proteins are labelled homologous when they carry the same Pfam domains in
the same order. Architecture is used rather than family membership because
identical domain content in a different order is a different protein, and rather
than gene names because those bind the metric to one annotation pipeline.

| condition | ‖μ‖ | isotropy | AUC | recall @5% FPR | precision@k | churn vs raw |
|---|---|---|---|---|---|---|
| raw | 0.9318 | 0.119 | 0.8796 | 0.581 | 0.302 | — |
| **random direction (control)** | **0.9317** | **0.119** | **0.8797** | **0.581** | **0.302** | **0.008** |
| r1 subtract μ | 0.0686 | 0.119 | 0.9218 | 0.709 | 0.317 | 0.328 |
| r2 project out μ̂ | 0.0621 | 0.119 | 0.9202 | 0.708 | 0.316 | 0.343 |
| abtt k=1 | 0.0784 | 0.160 | 0.9393 | 0.766 | 0.328 | 0.460 |
| abtt k=2 | 0.0770 | 0.178 | 0.9418 | 0.775 | 0.330 | 0.483 |
| abtt k=3 | 0.0598 | 0.200 | 0.9371 | 0.786 | 0.339 | 0.516 |
| soft-ZCA α=0.5 | 0.0513 | 0.522 | 0.9699 | 0.899 | 0.408 | 0.539 |
| **soft-ZCA α=0.1** | **0.0385** | **0.807** | **0.9712** | **0.912** | **0.429** | **0.595** |
| soft-ZCA α=0.01 | 0.0318 | 0.925 | 0.9672 | 0.901 | 0.437 | 0.618 |
| full PCA whitening | 0.0303 | 0.936 | 0.9653 | 0.897 | 0.438 | 0.623 |

Three things this establishes.

**The control behaves.** Projecting out a random direction reproduces raw to
four decimals and moves 0.4 of 50 neighbours. The effect is specific to the mean
direction, so it is anisotropy and not dimension removal.

**Dose matters, and the peak is interior.** AUC rises through soft-ZCA α=0.1 and
falls at full whitening, matching the published shape (single-direction removal
helps, untuned full-rank whitening costs). The optimum is the regularised
middle, so shipping full whitening as a default would give up 0.6 AUC points for
free.

**Two thirds of the ceiling is reachable.** Raw retrieval sits at AUC 0.880 and
recall 0.581; soft-ZCA α=0.1 reaches 0.971 and 0.912. Anchors feed chaining, so
this is upstream of every block and every cluster.

Isotropy tracks the dose (0.119 → 0.807) and confirms each transform did what it
claims, independent of whether it helped.

---

## Tier 1 — near-clonal cohorts (E. coli, cross-species)

Same ladder, hard negatives: each query's highest-scoring gene carrying a
*different* gene name, i.e. its most convincing impostor.

| corpus | condition | hard-neg AUC | precision@1 | cross-genus AUC | margin |
|---|---|---|---|---|---|
| E. coli / ESM2 | raw | 0.9974 | 0.998 | 0.983 | 0.021 |
| E. coli / ESM2 | soft-ZCA α=0.1 | 0.9970 | 0.998 | 0.983 | 0.296 |
| E. coli / ESM2 | full whitening | 0.9973 | 0.998 | 0.984 | 0.378 |
| cross-species / ESM2 | raw | 0.9865 | 0.987 | 0.954 | 0.020 |
| cross-species / ESM2 | soft-ZCA α=0.1 | 0.9870 | 0.988 | 0.954 | 0.279 |
| cross-species / ProtT5 | raw | 0.9851 | 0.992 | 0.955 | 0.127 |
| cross-species / ProtT5 | soft-ZCA α=0.1 | 0.9879 | 0.991 | 0.966 | 0.506 |

Raw retrieval on 20 near-clonal E. coli genomes is already at AUC 0.997 and
precision@1 0.998. There is no headroom, and the transform returns none. The
largest movement anywhere in this table is ProtT5 cross-genus AUC, +0.013.

### The margin column is the wrong instrument

Margin is a difference in raw cosine units, so any transform that spreads the
cosine distribution inflates it. On E. coli it grows 14× (0.021 → 0.296) while
AUC — rank-based, scale-free, measuring the same separation — moves −0.0004.
The margin widening is an axis rescaling.

Report AUC and precision@k. A scale-dependent statistic will confirm any
transform that changes scale, which is all of them.

---

## Tier 2 — full pipeline (DPANN, 3,564 genomes)

Thresholds are quantile-matched, so every arm retains the same fraction of kNN
similarities and the comparison is not confounded by keeping different numbers
of anchors.

| arm | threshold | blocks | clusters | minutes | peak RSS |
|---|---|---|---|---|---|
| raw | 0.8607 | 3,068,116 | 70,772 | 12.5 | 26.0 GB |
| abtt k=2 | 0.4559 | 3,211,383 | 72,932 | 13.0 | 28.8 GB |
| soft-ZCA α=0.1 | 0.2926 | 4,458,879 | 73,257 | 14.3 | 29.7 GB |

### Ribosomal-locus acceptance test

Maximal contiguous runs of ≥16 ribosomal-protein genes (Pfam `Ribosomal_*` as a
positional marker, which is an observed-domain criterion and not a system call).
Blocks covering ≥50% of a run, grouped by cluster.

| arm | ribosomal blocks | top-cluster purity | genomes | genera | Parvarchaeaceae recovered |
|---|---|---|---|---|---|
| raw | 4,397 | 52.3% | 143 | 8 | 94/94 |
| abtt k=2 | 4,990 | 45.6% | 144 | 9 | 94/94 |
| **soft-ZCA α=0.1** | **8,285** | **77.9%** | **148** | **10** | **94/94** |

Soft-ZCA finds 1.9× the ribosomal blocks and raises purity of the cluster
holding them from 52.3% to 77.9%. Genome recovery is complete in every arm, so
the transform improves what the cluster *contains* rather than which genomes
reach it. Cost is 45% more blocks and 1.8 minutes.

---

## Near-clonal block extent is a granularity property, not a retrieval error

On 20 E. coli genomes, blocks covering the ribosomal locus run a median 142
genes through a 26-gene locus. Prediction under test: better anchor
discrimination shortens them.

Swept in both spaces, measured on the blocks covering the ≥16-gene run:

| arm | threshold | ribosomal blocks | clusters | span median | span q90 | span max |
|---|---|---|---|---|---|---|
| raw | 0.85 | 182 | 21 | 142 | 284 | 2769 |
| raw | 0.95 | 182 | 21 | 142 | 284 | 2769 |
| raw | 0.99 | 183 | 21 | 107 | 278 | 2769 |
| soft-ZCA α=0.1 | 0.40 | 180 | 21 | 144 | 285 | 2769 |
| soft-ZCA α=0.1 | 0.60 | 182 | 21 | 142 | 284 | 2769 |
| soft-ZCA α=0.1 | 0.80 | 182 | 20 | 130 | 279 | 2769 |

**The prediction is refuted.** Span is flat across both spaces and every
threshold, and `span_max = 2769` is invariant throughout.

Long chains survive a 0.99 cosine cut, so they are built from high-confidence
anchors rather than marginal ones. In near-clonal genomes the 142-gene collinear
run is real: the chain passes through genuine orthologs in genuine order, and
there is no impostor to discriminate away. Raising the threshold deletes short
blocks and lifts the median by selection (16 → 21 over all blocks).

A locus boundary is visible where a cohort varies. Twenty near-identical genomes
supply nothing for a boundary to separate, so isolating a locus within them is a
question of cohort composition or of reporting sub-structure inside long blocks.
Embedding post-processing does not reach it.

---

## Relationship to the March 2026 result

`benchmarks/PCA_NORMALIZATION_PLAN.md` reports this ablation as run, with seven
treatments within ±0.3% and the conclusion that projection choice is immaterial.
That measurement rests on two broken foundations: it scored span-recall against
a hard-coded fallback list that a whole-contig baseline scores 100% on
(`docs/BENCHMARK_AUDIT_HANDOFF_2026-07-19.md`), and every row reports strict
recall of exactly 12.4%, the signature of the `.faa` coordinate defect.

Its conclusion holds on near-clonal bacterial cohorts and fails on divergent
ones. The reason is that all seven treatments were compared on a corpus with no
retrieval headroom, and generalised to corpora that have a great deal.

---

## How much of the available headroom does the treatment recover?

67 taxonomic strata of DPANN, from single species to the whole domain (9 to 2,500
genomes), scored under raw and soft-ZCA α=0.1. One transform fitted on held-out
genomes serves every stratum, so fit quality stays constant while difficulty
varies. Negatives are drawn inside each stratum. Raw AUC spans 0.909–0.986.

**Read the shared term first.** `gain = AUC_treated − AUC_raw` and
`headroom = 1 − AUC_raw` both carry `AUC_raw`, so regressing one on the other is
partly definitional — a treatment landing every stratum on one ceiling would fit
perfectly and mean nothing. Measured, `corr(AUC_raw, AUC_treated) = +0.924` with
a slope of +0.220, between a pure ceiling and a constant additive gain and closer
to a ceiling: spread contracts 4.2x, sd 0.0230 raw → 0.0055 treated.

The artifact-free quantity is the **fraction of headroom recovered**:

| Statistic | Value |
|---|---|
| Median | **0.785** |
| IQR | 0.762 – 0.812 |
| 95% CI (bootstrap over strata) | 0.771 – 0.793 |

| Rank | Strata | Median recovered |
|---|---:|---:|
| species | 12 | 0.787 |
| genus | 12 | 0.783 |
| family | 12 | 0.800 |
| order | 12 | 0.788 |
| class | 9 | 0.773 |
| phylum | 9 | 0.762 |
| whole domain | 1 | 0.755 |

Independent of every candidate predictor: headroom −0.088, isotropy −0.090, mean
norm +0.344, genome count +0.004.

**Soft-ZCA removes about four fifths of whatever retrieval error the raw space
carries**, whether that error is small or large. Note the mean norm's correlation
with the gain turns *negative* within one corpus (−0.399 across the 67 strata),
confirming it is a poor guide once model and dimension are held fixed.

---

## Does the fit transfer off its own corpus?

A transform is deployable when it survives leaving the corpus it was fitted on.
Tested by fitting on one archaeal **class** and evaluating on a disjoint one, so
the fit and evaluation sets share no genome. Soft-ZCA α=0.1, Pfam architecture
labels.

| eval clade | genomes | raw | fit in-corpus | fit on another clade | fit pooled-others |
|---|---|---|---|---|---|
| Nanobdellia | 1,738 | 0.9072 | 0.9781 | 0.9778–0.9781 | 0.9783 |
| Micrarchaeia | 1,146 | 0.9317 | 0.9851 | 0.9838–0.9842 | 0.9843 |
| Aenigmatarchaeia | 315 | 0.9185 | 0.9806 | 0.9798–0.9804 | 0.9805 |
| Iainarchaeia | 185 | 0.9267 | 0.9818 | 0.9806–0.9812 | 0.9814 |

(AUC. precision@50 tracks identically: 0.1933 in-corpus vs 0.1931–0.1946 across
fits on Nanobdellia.)

**The transform transfers.** A fit borrowed from a different archaeal class
performs within 0.001 AUC of one fitted in-corpus, and pooling several donor
clades matches or slightly beats the in-corpus fit. Fitting is a one-time cost
that can be paid on any reasonably related corpus.

The synthetic worry does not materialise at this distance. Soft-ZCA fitted on a
*deliberately unrelated* geometry does come back worse than raw, and that is
pinned in `tests/test_transform_wiring.py`. Between DPANN classes the geometries
are close enough that it never bites — which is the useful finding, since it
bounds where the caution applies.

### The threshold transfers too

Calibrated to 5% FPR on one clade, then applied as an **absolute cosine** to the
others — the way a shipped default is actually used.

| calibrated on | threshold | realized FPR elsewhere | realized recall |
|---|---|---|---|
| Nanobdellia | 0.1194 | 0.050 / 0.062 / 0.057 | 0.934–0.956 |
| Micrarchaeia | 0.1317 | 0.039 / 0.047 / 0.041 | 0.924–0.951 |
| Aenigmatarchaeia | 0.1281 | 0.042 / 0.053 / 0.043 | 0.927–0.952 |
| Iainarchaeia | 0.1196 | 0.050 / 0.061 / 0.057 | 0.934–0.956 |

Calibrated values cluster in 0.119–0.132, and any of them lands within 0.039–0.062
of the 5% target on every other clade.

### External 480-d arm

The plan's frozen-UniRef50 arm cannot run as written: UniRef50 was embedded at
480-d and the DPANN store is 320-d, so the frozen transform does not reach the
only corpus with headroom. Run for mechanics on the bacterial cohorts:

| corpus | fit | AUC | mean_norm | isotropy |
|---|---|---|---|---|
| E. coli | raw | 0.9989 | 0.9161 | 0.134 |
| E. coli | UniRef50 (external) | 0.9986 | 0.1427 | 0.622 |
| E. coli | in-corpus | 0.9987 | 0.0208 | 0.823 |
| cross-species | UniRef50 (external) | 0.9979 | 0.1441 | 0.622 |

An external fit isotropizes partially rather than fully (0.622 against 0.823),
and on saturated corpora that difference costs nothing. Measuring an external fit
against a divergent corpus needs UniRef50 re-embedded at 320-d.

---

## Calibrating `similarity_threshold`

`similarity_threshold` gates **kNN neighbours**, so it calibrates on that
population — every candidate is already a near neighbour, which makes the
negatives hard. A threshold set against random pairs answers a different
question and yields a different number (0.12 above, against 0.30–0.40 here).

Precision is the fraction of retained neighbours sharing a Pfam architecture;
retention is the fraction of the top-50 surviving. Mean over four clades.

| soft-ZCA threshold | retention | precision | raw precision at the same retention | ratio |
|---|---|---|---|---|
| 0.20 | 0.975 | 0.225 | 0.157 | 1.43× |
| 0.30 | 0.880 | 0.247 | 0.165 | 1.49× |
| 0.35 | 0.749 | 0.279 | 0.172 | 1.63× |
| 0.40 | 0.616 | 0.319 | 0.178 | 1.79× |
| 0.45 | 0.503 | 0.357 | 0.188 | 1.90× |
| 0.50 | 0.405 | 0.393 | 0.203 | 1.94× |
| 0.55 | 0.319 | 0.427 | 0.215 | 1.99× |
| 0.60 | 0.243 | 0.460 | 0.255 | 1.81× |

**In the transformed space the threshold works, and it roughly doubles anchor
precision at matched retention**, peaking near retention 0.3–0.5.

Raw space has no usable operating point on this curve:

| raw threshold | retention | precision | lift over keeping everything |
|---|---|---|---|
| 0.85 (current default) | 0.996 | 0.1546 | **+0.0005** |
| 0.90 | 0.976 | 0.1570 | +0.0030 |
| 0.95 | 0.815 | 0.1686 | +0.0146 |
| 0.98 | 0.265 | 0.2232 | +0.0691 |
| 0.99 | 0.024 | 0.5619 | +0.4079 |

The default retains 99.6% of neighbours for a precision lift of 0.0005, and the
first threshold that discriminates discards 97.6% of the anchor set. There is no
setting in between, which is the practical face of the anisotropy.

### Recommended value

**`similarity_threshold: 0.30` with soft-ZCA α=0.1.** It retains 88% of anchors
at 1.49× raw precision, and it is the operating point already validated
end-to-end: the Tier-2 DPANN run used the quantile-matched equivalent of 0.2926
and produced the ribosomal purity gain of 52.3% → 77.9%.

Values of 0.40–0.55 nearly double anchor precision and are **not** validated
downstream. Chaining needs enough surviving anchors per locus to reach
`min_chain_size`, so discarding 40–70% of them is a change that has to be
measured on blocks and clusters rather than inferred from this curve.

---

## Recommendation

Default stays `embedding_transform: none`. Enabling soft-ZCA α=0.1 by default
would change every stored vector, every threshold's meaning, and every
downstream number, on a benefit that is corpus-dependent and currently measured
on one divergent corpus.

Set `embedding_transform: soft_zca` with `transform_alpha: 0.1` for divergent
cohorts — MAGs, cross-phylum comparisons, anything where raw AUC sits well below
saturation. Leave it off for near-clonal cohorts, where it costs 45% more blocks
and returns nothing.

The threshold must be recalibrated with it: **`similarity_threshold: 0.30`**.
Absolute cosines are not comparable across spaces, and carrying 0.85 over would
retain essentially every neighbour.

Fitting can be borrowed. A pooled fit from related clades matches an in-corpus
fit within 0.001 AUC, so a shipped transform does not need to be refitted per
dataset — subject to the geometry caveat above.

### Resolved since the plan

- **The query path now enters the indexed space.** `elsa/search.py` applied no
  transform, so a query was scored against vectors from another space —
  well-formed cosines, confidently wrong neighbours. A dataset-fitted transform
  is now persisted to `ingest/embedding_transform.npz` at embed time and loaded
  at query time; a config that requests a transform against an index built
  without one raises rather than guessing.

- **Re-normalisation is not a confound.** The plan flagged five unconditional
  re-normalisation points as flattening magnitude-dependent transforms. ELSA
  retrieves with `METRIC_INNER_PRODUCT` over L2-normalised vectors, which is
  cosine, and cosine is invariant to per-vector scaling. Measured directly: with
  post-whitening norms spread 6.6–10.2, top-10 neighbour lists are identical and
  the largest cosine difference is exactly 0. No change needed.

- **Transfer and threshold calibration** — both answered above. A borrowed fit
  costs ≤0.001 AUC between archaeal classes, and `0.30` is the calibrated
  default.

### Open

- **External fit at 320-d.** UniRef50 exists only at 480-d, so the frozen arm
  ran on saturated bacterial cohorts where it cannot discriminate. Re-embedding
  UniRef50 at 320-d would test a genuinely external fit against a divergent
  corpus. Clade-disjoint transfer answers the same question with in-corpus data
  and makes this a confirmation rather than a dependency.

- **Thresholds above 0.30 are uncalibrated downstream.** 0.40–0.55 nearly double
  anchor precision on the retrieval curve. Whether that survives chaining — which
  needs enough anchors per locus to reach `min_chain_size` — is a Tier-2
  question, not a Tier-1 one.

- **Enabling the transform by default** still requires the near-clonal cost
  (45% more blocks for no gain) to be worth paying, or dispatch on a measured
  property of the corpus rather than a flag.

---

## Reproducing

```bash
# Tier 1, DPANN, domain-architecture labels
python benchmarks/scripts/ablate_embedding_transforms.py --label architecture

# Tier 1, near-clonal cohorts, hard negatives
python benchmarks/scripts/ablate_near_clonal.py

# Corpus sweep across ESM2/ProtT5 and dataset scale
python benchmarks/scripts/sweep_transform_corpora.py

# Does a fit transfer off its corpus? (clade-disjoint, plus the 480-d external arm)
python benchmarks/scripts/transfer_transform_fit.py --corpus dpann
python benchmarks/scripts/transfer_transform_fit.py --corpus bacterial

# Calibrate similarity_threshold on the kNN population it gates
python benchmarks/scripts/calibrate_similarity_threshold.py

# How much headroom does the treatment recover? (67 taxonomic strata)
python benchmarks/scripts/headroom_dose_response.py

# Is the gene-name label telling the truth? (identity vs cosine)
python benchmarks/scripts/validate_labels_by_identity.py
```

Results land in `benchmarks/evaluation/transform_ablation/`, with input hashes
and a claim ledger in `tier1_provenance.json`.
