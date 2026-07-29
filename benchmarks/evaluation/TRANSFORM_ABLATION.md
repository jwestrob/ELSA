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

## Recommendation

Default stays `embedding_transform: none`. Enabling soft-ZCA α=0.1 by default
would change every stored vector, every threshold's meaning, and every
downstream number, on a benefit that is corpus-dependent and currently measured
on one divergent corpus.

Set `embedding_transform: soft_zca` with `transform_alpha: 0.1` for divergent
cohorts — MAGs, cross-phylum comparisons, anything where raw AUC sits well below
saturation. Leave it off for near-clonal cohorts, where it costs 45% more blocks
and returns nothing.

The threshold must be recalibrated with it. Absolute cosines are not comparable
across spaces: quantile-matched thresholds here are 0.86 raw against 0.29
soft-ZCA, so carrying 0.85 across would retain nearly everything.

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

### Open

- **External fit arm.** All transforms are fitted in-corpus on a held-out split.
  The deployable form fits once on a reference corpus at the target dimension,
  and the generalisation gap is unmeasured.

  This arm carries a specific risk rather than only an unknown. Soft-ZCA
  suppresses the directions its fit corpus is spread along, so a corpus spread
  along different directions comes back **more** anisotropic than raw — measured
  on synthetic geometries and pinned in
  `tests/test_transform_wiring.py::test_whitening_fitted_on_a_different_geometry_makes_things_worse`.
  A frozen transform is safe on corpora resembling its fit corpus, so the
  external arm has to report the gap on a corpus deliberately unlike the fit set.

- **Threshold calibration.** Quantile matching makes arms comparable and does
  not produce a recommended default for the transformed space.

---

## Reproducing

```bash
# Tier 1, DPANN, domain-architecture labels
python benchmarks/scripts/ablate_embedding_transforms.py --label architecture

# Tier 1, near-clonal cohorts, hard negatives
python benchmarks/scripts/ablate_near_clonal.py

# Corpus sweep across ESM2/ProtT5 and dataset scale
python benchmarks/scripts/sweep_transform_corpora.py

# Is the gene-name label telling the truth? (identity vs cosine)
python benchmarks/scripts/validate_labels_by_identity.py
```

Results land in `benchmarks/evaluation/transform_ablation/`, with input hashes
and a claim ledger in `tier1_provenance.json`.
