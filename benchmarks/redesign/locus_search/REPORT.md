# Locus search and locus profiles on DPANN (ws_locus_search)

12 November 2026. ESM++ small vectors (embed_queue run `dpann__esmpp_small__20261111`),
permissive MMseqs2 tokens `e3c50s75`, DPANN package of 11 November. Written by the orchestrator
from the workstream agent's hand-back. The predeclared criterion is recorded in `bench.py`
(`PREDECLARED`) and in `out/criterion.json`, written by the first job before any number existed.

## Bottom line

1. **Predeclared criterion: not met.** Selection on odd genomes picked the iterative profile
   (M3 grow, round 3). On even-genome DefenseFinder queries it gains +0.245 mean AP
   [+0.235, +0.255] over tokens (M0) and makes 2.81 unlabelled hits per query at E ≤ 0.01 against
   0.72 for M0 (3.9×; the bound was 1.5×).
2. **Tokens plus embedding ranks (M2) would pass**: +0.206 AP [+0.198, +0.213] at 0.92
   unlabelled hits per query (1.27×). It was not the selected method.
3. **The gain comes from graded per-gene evidence, and sequence search supplies it better.**
   Replacing embedding ranks by MMseqs2 bitscore ranks (E ≤ 10, top 20,000 per gene): tokens +
   sequence rank (S2) beats M2 by 0.044 AP [0.036, 0.052]; sequence rank alone (S1) beats
   embedding rank alone (M1) by 0.160.
4. **Embeddings add ranking value on top of sequence search.** S3 (tokens + sequence rank +
   0.5 × embedding rank) minus S2: +0.079 AP [+0.074, +0.083], +0.047 strict AP, +0.08
   unlabelled hits per query. Recall at E ≤ 0.01 moves by +0.002: the embedding reorders the
   deeper list.
5. **Targets invisible to tokens are rarely retrieved by embeddings.** Recall gain on
   token-invisible targets at E ≤ 0.01 against M0: M2 +0.003 [+0.001, +0.006], M1 +0.017, M3 grow
   round 3 +0.058, S2 +0.050. Phase 0 counted 120,946 even-genome pairs as token-invisible and
   embedding-reachable; M1 retrieves 3,023 of them at E ≤ 0.01 (2.5%), M2 116.
6. **Profile library (phase 2).** Token profiles built from odd-genome engine modules annotate
   even genomes without recomputing recurrence: tighter loci than the engine (DefenseFinder
   excess +0.061) at lower F1 (−0.131) and lower coverage.

## Method

- Evidence for a query gene and a database gene: −log of the fraction of database genes that
  match at least as well. Tokens use family size; embeddings use the cosine tail rank (exact,
  K = 20,000, own genome excluded).
- A hit is a window of at most W genes on one contig. Each target gene fills one query slot
  (greedy). Score = sum over filled slots of (evidence − log window span) for slots at or above
  a floor τ. Overlapping windows are resolved by score.
- E-values: 200 replicates of a null that circularly shifts each slot's evidence track
  independently along the genome-ordered gene array.
- Labels: correct = covers ≥ 50% of the members of a same-type target with ≥ 2 members; wrong =
  overlaps a member of a different-type instance; unlabelled = overlaps no instance of the
  caller. Primary AP ignores unlabelled hits; strict AP counts them wrong.
- Tuned on odd `genome_code`, tested on even; genome-bootstrap 95% intervals (2,000 draws).
- Types: `system_type` for DefenseFinder and TXSScan; `system_subtype` for CRISPRCasTyper (its
  `system_type` is the string "False" for 3,389 of 3,820 instances).
- Tuned values (DefenseFinder, odd genomes): M0 window = query span + 8, τ 3; M1 span + 8, τ 2;
  M2 weight 0.5, span + 8, τ 0. Going from +4 to +8 genes changes AP by ≤ 0.002.

## Phase 0: headroom

Pairs of same-type instances with ≥ 2 members in different genomes. Embedding-reachable = best
cosine ≥ 0.977 (99.9th percentile of 2 M random gene pairs; their median is 0.78) or the target
gene is in the query gene's top 0.1% by rank.

| Caller | Ordered pairs | Token-reachable | Embedding-reachable | Either | Token-invisible, embedding-reachable |
|---|---|---|---|---|---|
| DefenseFinder | 1,146,374 | 25.6% | 37.5% (cosine 8.8%, rank 37.2%) | 47.3% | 20.6% |
| Control: 200,000 different-type pairs | | 0.33% | 7.5% | 7.7% | 7.4% |
| TXSScan (one type) | 645,216 | 21.5% | 25.8% | 39.5% | 11.9% |
| CRISPRCasTyper | 3,571,536 | 8.7% | 8.0% | 11.7% | 2.9% |
| Control: different-type pairs | | 2.3% | 2.9% | 4.0% | 1.7% |

Embedding headroom above the different-type floor is about 13 points for DefenseFinder; part of
the floor is homology shared across restriction-modification types.

## Phase 1: DefenseFinder, even genomes, ≥ 2 members (1,247 queries)

| Method | MAP | Strict MAP | Recall E≤0.01 | Recall E≤1 | Token-invisible recall E≤0.01 / E≤1 | Unlabelled per query E≤0.01 | Wrong per query E≤0.01 |
|---|---|---|---|---|---|---|---|
| M0 tokens | 0.370 | 0.256 | 0.094 | 0.173 | 0.000 / 0.000 | 0.72 | 0.11 |
| M1 embedding | 0.442 | 0.257 | 0.093 | 0.171 | 0.017 / 0.073 | 2.59 | 0.33 |
| M2 tokens + 0.5 × embedding | 0.576 | 0.362 | 0.115 | 0.233 | 0.003 / 0.031 | 0.92 | 0.21 |
| M3 refine, round 2 / 3 | 0.600 / 0.611 | 0.384 / 0.398 | 0.147 / 0.169 | 0.262 / 0.282 | 0.024 / 0.050 (E≤0.01) | 1.24 / 1.49 | 0.96 / 1.39 |
| M3 grow, round 2 / 3 (selected) | 0.604 / 0.615 | 0.386 / 0.399 | 0.149 / 0.174 | 0.265 / 0.287 | 0.028 / 0.058 (E≤0.01) | 1.87 / 2.81 | 1.13 / 1.68 |
| S1 sequence rank (control) | 0.602 | 0.389 | 0.170 | 0.291 | 0.046 / 0.131 | 4.47 | |
| S2 tokens + sequence rank | 0.620 | 0.407 | 0.180 | 0.305 | 0.050 / 0.143 | 4.44 | |
| S3 = S2 + 0.5 × embedding | **0.698** | **0.454** | 0.182 | 0.320 | 0.052 / 0.160 | 4.51 | |

- Instances with ≥ 3 members keep the ordering: M0 0.405, M2 0.653, M3 refine round 3 0.750, S2
  0.705, S3 0.780.
- M3 drift: recall at E ≤ 0.01 rises from 0.115 (M2) to 0.169–0.174 by round 3 while wrong-type
  hits per query rise 5–8× (0.21 to 0.96–1.68); primary precision 0.979 → 0.963, strict precision
  0.882 → 0.834–0.867.
- Window width is not the source of the gain: correct hits at E ≤ 0.01 have median boundary
  precision 1.0 for every method (mean 0.81 M0, 0.80 M1 and M2).
- M2 finds 6,943 targets that M0 misses (6,208 of them RM_Type_I; 138 token-invisible): targets
  that tokens see and score too weakly, recovered by graded per-gene ranks.
- Per-type MAP (macro over 36 types): M0 0.630, M1 0.563, M2 0.740. RM_Type_II (431 queries):
  0.126 / 0.202 / 0.284.
- TXSScan: MAP 0.221 (M0), 0.581 (M2), 0.732 (M3 grow round 3), with 65 / 107 / 257 unlabelled
  hits per query at E ≤ 0.01 and a single type. CRISPRCasTyper: 0.091 (M0), 0.116 (M2), 0.123
  (M3 grow round 3); M1 is 0.008 below M0.

## Phase 2: profile library

The engine was rerun on odd genomes only (job 78273; rule P ≤ 1e-4 and L ≥ 2; 91,522
significant pairs; 1,287 modules over 26,104 tokens; 47 s). Profiles have a median of 2 slots
(largest 345) and a median window of 3. They are scanned over the 1,782 even genomes (1,797,111
genes) and scored with the package scorer against the full engine's loci on those genomes.

| Token profiles; locus = ≥ 2 slots of one module co-located | Profile F1 | Engine F1 | ΔF1 | Δ excess over random | Covered ≥ 50%: profile / engine |
|---|---|---|---|---|---|
| DefenseFinder (1,261) | 0.675 | 0.806 | −0.131 [−0.147, −0.113] | +0.061 [+0.045, +0.078] | 0.773 / 0.906 |
| TXSScan (404) | 0.531 | 0.758 | −0.227 | −0.067 | 0.728 / 0.995 |
| CRISPRCasTyper (1,899) | 0.371 | 0.589 | −0.219 | −0.036 | 0.637 / 0.887 |

- Requiring a per-hit E ≤ 1 drops DefenseFinder F1 to 0.310. Module reach is not the limit: 94%
  of even-genome DefenseFinder instances have at least half their members in one odd-genome
  module.
- Combined token + embedding profiles with the ≥ 2-slot rule fill slots almost everywhere (5.67 M
  loci spanning 11× the even-genome gene count; F1 0.862, excess 0.103): a coverage artefact.
- Speed: token library 298 genomes per second (64 CPUs, including the null); combined library 9
  genomes per second.

## Speed

- Exact embedding top-K: 98 vectors per second on 64 CPUs (3.58 M × 960, K = 20,000).
- Per query, scan plus 200-replicate null: M0 0.38 CPU-s; M1 and M2 about 10.4 CPU-s (median
  7.5); M3 11–13.5 CPU-s. The scan alone costs 0.045 CPU-s; the null dominates.
- MMseqs2 search for the sequence-rank control: 11 min, 44.5 M hits.

## Open

1. The unlabelled hits of the sequence-rank and M3 methods (2–6× M0) are uninspected.
2. With the 50% rule a 2-member query is satisfied by one gene, so MAP partly measures
   single-gene ranking; the ≥ 3-member stratum gives the same ordering.
3. Tuning maximized MAP without a rate constraint, which is why M3 grow was selected.
4. ESM++ small is the weaker model on the Spacedust merges; ESM2-650M could change S3 − S2.
5. A library of the odd-genome significant token pairs is the next phase-2 design.

## Jobs and files

Jobs (all completed): 78166 (top-K, phase 0, tuning), 78183 (all callers, M0–M3), 78273 (phase 2
and phase-0 control), 78290 (phase 2 with the ≥ 2-slot rule), 78292 (DefenseFinder round 1, tag
`_dfr1`), 78316 (sequence-rank control S1, S2), 78347 (S3).

Local `benchmarks/redesign/locus_search/`: `locus_search.py` (`search(query_gids, …)`),
`bench.py`, `phase2.py`, `seq_control.py`, `phase0_control.py`, `posthoc.py`, `run_ls.sbatch`,
`tables.py` (`tables.py results` prints every table), `selftest*.py`, `results/`.

Biotite `redesign_20261111/locus_search/out/`: `summary.json`, `final_metrics.parquet`,
`final_hits_E10.parquet` (11.6 M hits), `pairs.parquet`, `summary_seqctl.json`,
`summary_seqctl3.json`, `phase2/phase2_summary_v2.json`; logs `logs/locus_search_<jobid>.log`.
