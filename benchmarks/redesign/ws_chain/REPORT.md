# ws_chain report

Copied verbatim by the coordinator from the closing entries of `ws_chain/NOTEBOOK.md` (line 407 to the end); the agent's own REPORT.md write was blocked by the harness.

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
