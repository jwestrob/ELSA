# Task headroom, more than representation geometry, governs the value of embedding post-processing in protein language model retrieval

**Draft — August 2026.** Analysis code and result tables are listed under
[Data and code](#data-and-code).

---

## Abstract

Mean-pooled protein language model (pLM) embeddings occupy a narrow cone: on a
3.4-million-protein archaeal corpus the mean of the L2-normalised embeddings has
norm 0.932, so a shared direction accounts for 87% of every vector and unrelated
proteins sit at a median cosine of 0.935. Text-embedding work reports that
removing this component improves retrieval, with the gain tracking the mean norm.
We ran the corresponding dose-response ablation for protein retrieval — eleven
treatments from single-direction removal through regularised and full whitening,
with a random-direction control — across seven corpus and representation
combinations spanning three pLMs and mean norms from 0.028 to 0.932, scored
against ordered Pfam domain architecture.

Post-processing produced large gains where retrieval had room to improve: on two
independent divergent archaeal and bacterial corpora, soft-ZCA whitening raised
homology-detection AUC from 0.888 to 0.972 and from 0.896 to 0.968, and raised
recall at 5% FPR from 0.587 to 0.912. On near-clonal cohorts the same treatments
returned AUC differences of 0.0005 or less, because raw retrieval already reached
0.998. Across all seven combinations the improvement was predicted almost
entirely by the headroom available (r = +0.999 against 1 − AUC_raw), while the
mean norm predicted it weakly (r = +0.477) and the covariance participation ratio
hardly at all (r = +0.055). Two PCA-projected corpora, whose mean component was
already removed, still benefited from whitening — locating the remaining signal
in the covariance spectrum instead of the first moment.

The optimum was interior: regularised whitening at shrinkage α = 0.1 outperformed
full-rank whitening on every divergent corpus. Fitted transforms transferred
between disjoint archaeal classes within 0.001 AUC, so one fit serves related
corpora. Downstream, the treatment improved syntenic cluster composition on a
3,564-genome corpus, raising the ribosomal-protein content of the cluster holding
that locus from 52.3% to 77.9%. Syntenic block extent in near-clonal cohorts
stayed fixed across every treatment and threshold, identifying it as a property
of cohort composition.

---

## 1. Introduction

Protein language models encode sequences into vectors whose cosine similarity is
used directly for homology search, annotation transfer, and — the setting here —
synteny discovery, where each gene retrieves neighbours to seed collinear chains.
That pipeline inherits whatever geometry the embedding space has.

Contextual embedding spaces are known to be anisotropic: representations
concentrate in a narrow cone so that arbitrary pairs score highly. In natural
language processing this motivated a family of corrections — subtracting the
corpus mean, projecting out dominant directions (all-but-the-top), and whitening
the covariance. Recent work assembles these into a ladder and reports that
single-direction removal yields consistent gains of 0.6–0.9 points while
full-rank whitening degrades every model tested, with benefit tracking the mean
norm at r = +0.72 across ‖μ‖ ∈ [0.19, 0.85] (arXiv 2511.11041). Regularised
whitening occupies the middle and has been applied to code search (ESANN 2025)
and dense passage retrieval (arXiv 2209.00218), with counter-evidence for
classification (RepL4NLP 2024) and a proposed mechanism for whitening's failures
in heavy-tailed spectra (arXiv 2603.19339).

Protein work has adopted the diagnostic vocabulary — IsoScore appears in recent
pLM benchmarking — while the corresponding dose-response ablation for protein
retrieval remains open. Two features make the protein case worth measuring
rather than assuming. Mean-pooled pLM embeddings reach mean norms above the
entire range tested in the text literature, so applying the published slope means
extrapolating past its support. And protein retrieval spans an unusually wide
difficulty range: the same method serves near-identical laboratory isolates and
remote homologs across phyla, which lets us ask what actually sets the size of
the benefit.

We report that ablation, and find that the governing variable is how much
retrieval error exists to remove.

---

## 2. Methods

### 2.1 Corpora

Seven corpus and representation combinations, chosen to span both taxonomic
divergence and pLM architecture.

| Corpus | Representation | Dim | Genomes | Labelled proteins |
|---|---|---:|---:|---:|
| DPANN archaea | ESM2-t12 | 320 | 3,564 | 18,362 |
| Omnitrophota | ESM2-t12 | 320 | — | 23,616 |
| *E. coli* | ESM2-t12 raw | 480 | 20 | 24,543 |
| *E. coli* | ESM2-t12 + PCA | 256 | 20 | 24,505 |
| Cross-species | ESM2-t12 raw | 480 | 30 | 21,111 |
| Cross-species | ProtT5-XL-U50 | 1024 | 30 | 21,111 |
| Cross-species | gLM2-150M + PCA | 256 | 30 | 20,815 |

The cross-species cohort mixes *Escherichia*, *Salmonella* and *Klebsiella*.
DPANN and Omnitrophota are metagenome-assembled genome collections spanning
phylum-level diversity. Genes were called with Prodigal from assemblies so that
contig identifiers and genomic coordinates are preserved.

### 2.2 Homology labels

Two proteins are labelled homologous when they carry the **same Pfam domains in
the same order along the sequence**. Ordered architecture is the protein-level
identity signal: a shared top domain would equate an A+B protein with a bare A,
and Pfam clans group related families too coarsely to mean "the same protein".

This label is HMM-derived and therefore independent of both the embedding under
test and of alignment-based search, which allows it to judge remote-homology
claims that a sequence-identity label would decide circularly. It is also
independent of the pipeline the embeddings feed, so it carries no dependence on
the synteny benchmark whose separate limitations we discuss in §3.7.

### 2.3 Treatment ladder

Given embeddings **X** with mean **μ** and covariance **Σ**, all treatments were
fitted on held-out rows and applied to the evaluation rows.

| Condition | Operation |
|---|---|
| raw | L2 normalisation alone |
| r1_center | subtract **μ** |
| r2_project | project out **μ̂** |
| **random_direction** | **project out an arbitrary unit direction (control)** |
| abtt_k | subtract **μ**, project out the top *k* principal directions, k ∈ {1,2,3} |
| soft_zca_α | ZCA whitening of the shrunk covariance (1−α)**Σ** + α·tr(**Σ**)/d·**I**, α ∈ {0.5, 0.1, 0.01} |
| pca_whiten | full-rank PCA whitening |

The random-direction control separates "the mean direction carries the problem"
from "removing any direction helps". It should score like raw; a control that
instead scores like the mean-removal arms would void the table.

### 2.4 Metrics

**AUC** and **recall at a matched 5% false-positive rate** over positive pairs
(same architecture) against negatives. **Precision@50** — the fraction of each
protein's fifty nearest neighbours sharing its architecture — is the operational
endpoint, because the top-50 *is* the anchor set the synteny pipeline consumes.
**Mean norm** ‖mean of unit vectors‖ and an **isotropy** score (participation
ratio of the normalised covariance spectrum) verify that each treatment did what
it claims.

Rank-based statistics are used throughout. A margin — the raw cosine gap between
a positive and its hardest negative — is scale-dependent, so any treatment that
spreads the cosine distribution inflates it mechanically. On *E. coli* the margin
grows 14-fold under soft-ZCA (0.021 → 0.296) while AUC moves −0.0004. We report
AUC and precision@50 for this reason.

Thresholds are quantile-matched across conditions wherever conditions are
compared, since an absolute cosine means different things in different spaces.
§3.6 treats threshold selection as its own question, where quantile matching is
deliberately set aside.

### 2.5 Leakage protocol

Transforms were fitted on a held-out **genome** split and evaluated on the
remainder, so no transform sees its test data. Fit rows were sampled randomly
across the fit split: a contiguous head slice draws a handful of related genomes
whose covariance is rank-deficient, which silently destabilises whitening. §3.5
strengthens this to a clade-disjoint split.

---

## 3. Results

### 3.1 pLM embedding spaces are severely anisotropic

On DPANN (3,395,472 vectors, 320-d, L2-normalised):

| Quantity | Value |
|---|---|
| ‖mean of unit vectors‖ | 0.9318 |
| Isotropy (participation ratio) | 0.1187 |
| Median cosine, unrelated proteins | 0.935 |
| Median cosine, anchored pairs | 0.996 |

A shared direction accounts for 87% of every embedding, and the discriminative
signal occupies the last 6% of the cosine range. A practical consequence is
visible in the pipeline this corpus feeds: raising its similarity threshold from
0.85 to 0.95 removes 0.9% of retained anchors, so the parameter is a threshold in
name while the geometry keeps everything above it.

Mean norms across the seven combinations span 0.028 to 0.932, giving the
comparison in §3.4 real leverage.

### 3.2 The dose-response optimum is interior

Full ladder on DPANN, ordered Pfam architecture labels, quantile-matched
thresholds:

| Condition | ‖μ‖ | Isotropy | AUC | Recall @5% FPR | Precision@50 | Neighbour churn |
|---|---:|---:|---:|---:|---:|---:|
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

AUC rises monotonically through regularised whitening at α = 0.1 and falls at
full rank, reproducing the shape reported for text embeddings: single-direction
removal helps, and untuned full whitening gives back part of the gain. Shipping
full whitening as a default would forfeit 0.6 AUC points.

Raw retrieval sits at AUC 0.880 and recall 0.581; the best treatment reaches
0.971 and 0.912, recovering roughly two-thirds of the available headroom.
Precision@50 rises from 0.302 to 0.429, and since the top-50 is the anchor set,
this propagates to every block and cluster downstream (§3.7).

Isotropy tracks the dose (0.119 → 0.807), confirming each treatment performed its
stated operation independently of whether it helped.

### 3.3 The effect belongs to the mean direction

Projecting out a random unit direction reproduces raw to four decimal places
(AUC 0.8797 against 0.8796) and moves 0.4 of every 50 retrieved neighbours. The
gain therefore belongs to the specific direction removed rather than to dimension
removal in general, which is the control the ladder needs to be readable.

### 3.4 Task headroom predicts the benefit; representation geometry predicts it weakly

Raw and best-treatment AUC across all seven combinations:

| Corpus | Representation | ‖μ‖ raw | Isotropy raw | AUC raw | AUC soft-ZCA | Gain |
|---|---|---:|---:|---:|---:|---:|
| DPANN | ESM2-320 | 0.9318 | 0.118 | 0.8877 | 0.9718 | **+0.0841** |
| Omnitrophota | ESM2-320 | 0.9190 | 0.144 | 0.8964 | 0.9676 | **+0.0712** |
| *E. coli* | ESM2-480 | 0.9155 | 0.133 | 0.9984 | 0.9989 | +0.0005 |
| Cross-species | ESM2-480 | 0.9162 | 0.132 | 0.9981 | 0.9985 | +0.0004 |
| Cross-species | ProtT5-1024 | 0.6199 | 0.136 | 0.9979 | 0.9986 | +0.0007 |
| Cross-species | gLM2-256 | 0.0999 | 0.069 | 0.9936 | 0.9971 | +0.0035 |
| *E. coli* | ESM2-PCA256 | 0.0284 | 0.155 | 0.9981 | 0.9989 | +0.0008 |

Correlations of the gain against candidate predictors:

| Predictor | r |
|---|---:|
| **Headroom (1 − AUC_raw)** | **+0.999** |
| Mean norm ‖μ‖ | +0.477 |
| Isotropy | +0.055 |

The two divergent corpora — an archaeal and a bacterial MAG collection,
independently assembled and annotated — replicate each other closely (+0.084 and
+0.071). The five near-clonal or near-saturated combinations return gains an
order of magnitude smaller, and they do so at mean norms spanning almost the
whole measured range, including two combinations whose ‖μ‖ (0.916, 0.920) matches
the divergent corpora almost exactly.

Anisotropy is therefore a condition for post-processing to have something to
correct, while the size of the correction is set by how much retrieval error
exists. In a saturated regime, geometry that looks equally pathological yields
almost nothing, because the ranking it distorts is already right.

**On the strength of r = +0.999.** The seven points form two clusters — two
divergent corpora and five saturated ones — so this coefficient measures
separation between regimes rather than a calibrated slope within either. The
defensible claim is ordinal: headroom orders the gains correctly and the geometry
statistics do not. Establishing a slope requires corpora sampled across
intermediate difficulty, which we identify as the natural next experiment.

### 3.4.1 The residual signal lives in the covariance, not the mean

Two combinations carry a PCA projection that has already removed the mean:
*E. coli* ESM2-PCA256 (‖μ‖ = 0.028) and cross-species gLM2-256 (‖μ‖ = 0.100).
Both retain low isotropy (0.155, 0.069) and both still improve under whitening
(+0.0008, +0.0035) — the larger of the two gains among saturated corpora
belonging to gLM2, the most nearly isotropic space by mean norm in the study.

Mean norm captures a first moment. What the successful treatments repair is the
shape of the covariance spectrum, which a centred space can retain in full. This
reframes the mean-norm diagnostic prevailing in the text literature as a partial
proxy, and explains why regularised whitening — which acts on the spectrum —
outperforms every direction-removal arm in §3.2.

### 3.5 A fitted transform transfers between clades

Fitted on one archaeal class and evaluated on a disjoint one, sharing no genome:

| Evaluation clade | Genomes | raw | Fit in-corpus | Fit on another clade | Fit pooled |
|---|---:|---:|---:|---:|---:|
| Nanobdellia | 1,738 | 0.9072 | 0.9781 | 0.9778–0.9781 | 0.9783 |
| Micrarchaeia | 1,146 | 0.9317 | 0.9851 | 0.9838–0.9842 | 0.9843 |
| Aenigmatarchaeia | 315 | 0.9185 | 0.9806 | 0.9798–0.9804 | 0.9805 |
| Iainarchaeia | 185 | 0.9267 | 0.9818 | 0.9806–0.9812 | 0.9814 |

A borrowed fit lands within 0.001 AUC of an in-corpus fit, and pooling donor
clades matches or slightly exceeds it. Precision@50 tracks identically
(0.1933 in-corpus against 0.1931–0.1946 borrowed). Fitting is therefore a
one-time cost payable on any reasonably related corpus.

The transferability has a boundary. Soft-ZCA suppresses the directions its fit
corpus is spread along, so a corpus spread along different directions returns
*more* anisotropic than raw; we reproduce this on synthetic geometries with
disjoint dominant axes. Between archaeal classes the geometries stay close enough
that transfer holds, which bounds where the caution applies.

### 3.6 Threshold calibration belongs to the retrieved population

A similarity threshold in this pipeline gates **k-nearest neighbours**, where
every candidate is already a near neighbour and the negatives are correspondingly
hard. Calibrating against random pairs answers a different question and returns a
different number (0.12 against 0.30 here).

Precision is the fraction of retained top-50 neighbours sharing an architecture;
retention is the fraction of the top-50 surviving. Mean over four clades:

| soft-ZCA threshold | Retention | Precision | Raw precision at equal retention | Ratio |
|---:|---:|---:|---:|---:|
| 0.20 | 0.975 | 0.225 | 0.157 | 1.43× |
| 0.30 | 0.880 | 0.247 | 0.165 | 1.49× |
| 0.40 | 0.616 | 0.319 | 0.178 | 1.79× |
| 0.50 | 0.405 | 0.393 | 0.203 | 1.94× |
| 0.55 | 0.319 | 0.427 | 0.215 | 1.99× |

The treated space roughly doubles anchor precision at matched retention, peaking
near retention 0.3–0.5. In the raw space the same sweep shows why a threshold
there is inert: at 0.85 it retains 99.6% of neighbours for a precision lift of
0.0005, and the first value that discriminates (0.99) retains 2.4%. The parameter
becomes usable only after the space is treated.

Calibrated thresholds also transfer: set to 5% FPR on one clade and applied as an
absolute cosine to the others, realised FPR lands in 0.039–0.062 against a 0.05
target, with calibrated values clustering in 0.119–0.132.

### 3.7 Downstream effect on syntenic clustering

Full pipeline on DPANN (3,564 genomes, 3.4M genes), quantile-matched thresholds
so each arm retains the same fraction of similarities. Loci were defined as
maximal runs of ≥16 ribosomal-protein genes admitting gaps of up to three gene
positions, identified by Pfam `Ribosomal_*` domains used as a positional marker.
The domain hit is treated as a coordinate, carrying no claim about the ribosome
as an assembled system.

| Arm | Threshold | Blocks | Ribosomal blocks | Top-cluster purity | Qualifying genomes in that cluster |
|---|---:|---:|---:|---:|---|
| raw | 0.8607 | 3,068,116 | 4,397 | 52.3% | 94/94 |
| abtt k=2 | 0.4559 | 3,211,383 | 4,990 | 45.6% | 94/94 |
| **soft-ZCA α=0.1** | 0.2926 | 4,458,879 | **8,285** | **77.9%** | 94/94 |

The treatment nearly doubles the ribosomal blocks recovered and raises the
ribosomal content of the cluster holding that locus from 52.3% to 77.9%. Genome
recovery is complete in every arm, so the improvement lies in what the cluster
contains rather than which genomes reach it. The cost is 45% more blocks and
about two minutes of additional runtime.

### 3.8 Block extent in near-clonal cohorts is set by cohort composition

In 20 near-clonal *E. coli* genomes, blocks covering a ≥16-gene ribosomal locus
run a median 142 genes through a locus whose own median span is 26 genes. We tested whether improved anchor
discrimination tightens them, sweeping the threshold in both spaces and measuring
the blocks covering that locus:

| Arm | Threshold | Blocks | Span median | Span q90 | Span max |
|---|---:|---:|---:|---:|---:|
| raw | 0.85 | 182 | 142 | 284 | 2769 |
| raw | 0.99 | 183 | 107 | 278 | 2769 |
| soft-ZCA | 0.40 | 180 | 144 | 285 | 2769 |
| soft-ZCA | 0.80 | 182 | 130 | 279 | 2769 |

Span holds steady across both spaces and every threshold, with an invariant
maximum. Chains surviving a 0.99 cosine cut are built from high-confidence
anchors, and in near-identical genomes a 142-gene collinear run is genuine: the
chain passes through orthologs in their true order. A locus boundary becomes
visible where a cohort varies, so resolving one is a matter of cohort composition
or of reporting sub-structure within long blocks. Embedding post-processing
addresses a different problem.

---

## 4. Discussion

The practical rule this supports is to select the treatment from the difficulty
of the retrieval task. Divergent collections — MAGs, cross-phylum comparisons,
anything where raw retrieval sits well below saturation — gain substantially from
regularised whitening at α = 0.1, together with a threshold recalibrated for the
treated space. Near-clonal cohorts retain their raw representation, where the
treatment costs additional blocks and returns a difference within noise.

Two findings refine the account inherited from text embeddings. Benefit tracking
mean norm holds within a family of models on a fixed task; across tasks of
differing difficulty, headroom dominates, and two corpora with nearly identical
‖μ‖ (0.916 and 0.919) sit at opposite ends of the gain distribution.
Complementarily, spaces whose mean has already been removed by PCA still improve
under whitening, placing the operative structure in the covariance spectrum. A
diagnostic that measures only the first moment will therefore mark some
correctable spaces as already healthy.

The interior optimum reproduces across every divergent corpus we tested,
supporting the recommendation against untuned full-rank whitening as a default
and extending it from text retrieval to protein retrieval.

For synteny specifically, the gain reaches the cluster level: anchors feed
chaining, and higher anchor precision produced measurably cleaner clusters on a
3,564-genome corpus. The limits are equally clear. Block extent in near-clonal
cohorts is a property of genuine collinearity, and no treatment of the embedding
space alters it.

### Limitations

The headroom relationship rests on seven corpus-representation combinations
forming two clusters, which supports an ordinal claim and leaves the functional
form open. Ordered Pfam architecture, while independent of the embedding and of
alignment search, is available only for annotated proteins and inherits Pfam's
coverage; domain-dark proteins are absent from the labelled set. Downstream
evaluation covers one pipeline and one locus definition. The external-fit arm ran
at 480 dimensions against corpora whose retrieval was already saturated, since
the available reference embedding set and the divergent corpora differ in
dimension; clade-disjoint transfer addresses the same question with in-corpus
data.

### Data and code

| Artifact | Path |
|---|---|
| Treatment ladder | `elsa/transform.py` |
| Tier-1 driver | `benchmarks/scripts/ablate_embedding_transforms.py` |
| Near-clonal hard negatives | `benchmarks/scripts/ablate_near_clonal.py` |
| Corpus sweep | `benchmarks/scripts/sweep_transform_corpora.py` |
| Transfer and external fit | `benchmarks/scripts/transfer_transform_fit.py` |
| Threshold calibration | `benchmarks/scripts/calibrate_similarity_threshold.py` |
| Result tables and provenance | `benchmarks/evaluation/transform_ablation/` |
| Full internal report | `benchmarks/evaluation/TRANSFORM_ABLATION.md` |

### References

1. Correcting Mean Bias in Text Embeddings. arXiv:2511.11041.
2. Isotropy Matters: Soft-ZCA Whitening for Semantic Code Search. ESANN 2025 / arXiv:2411.17538.
3. Isotropic Representation Can Improve Dense Retrieval. arXiv:2209.00218.
4. Whitening Not Recommended for Classification Tasks. RepL4NLP 2024.
5. Spectral Tempering for Dense Passage Retrieval. arXiv:2603.19339.
6. Mu & Viswanath. All-but-the-Top: Simple and Effective Post-processing for Word Representations.
7. Su et al. 2021. Whitening Sentence Representations for Better Semantics and Faster Retrieval.
