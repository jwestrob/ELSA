# ws_seqanchors report: sequence anchors, embedding anchors and their union through one chainer

11 November 2026. Spacedust collection: 67,148 query proteins (9 actinomycete genomes)
against 4,190,383 reference proteins (1,308 genomes). Notebook: `NOTEBOOK.md` (predeclared
criteria at 03:15, before any result). Remote: `$RD/ws_seqanchors/`.

## Findings

1. **Sequence anchors carry a clean null without any context selection.** Pairwise MMseqs2
   anchors chain to blocks whose permuted/observed ratio is 0.005 (top 60 per query) or
   0.033 (every hit). Every block passes the 5% calibration. Context60 embedding anchors run
   at 0.67-0.93, and the 5% cut has to discard most of their 2-3-anchor blocks.
2. **Exhaustive sequence anchors (`S-all`) are the strongest arm on the benchmark and on
   distant genomes.** F1 excess over random placement is +0.073 [+0.052, +0.094] (the
   highest of 24 arms), with 3.68 M accepted blocks, 2.49 M of them cross-phylum and 16,975
   cross-phylum blocks of >= 10 anchors. The best context60 embedding arm finds 17,544
   cross-phylum blocks (51 of >= 10 anchors).
3. **No embedding or hybrid arm beats a sequence arm on `source207`.** Paired region
   bootstrap: every embedding-only arm is below `S-all` (by 0.025 to 0.094); context60
   hybrids are below or equal to `S-top60`; seed hybrids equal it (+0.001 to +0.002, intervals
   within +-0.017).
4. **The predeclared primary criterion is met, narrowly, and for a reason that favours
   sequence.** Against the budget-matched `S-top60/bitrank`, `E-ctx60/rank` adds 63.9%
   (ESM2-650M) and 30.6% (ESM-C 600M) accepted pairs and 32.1% / 25.2% covered query genes,
   and 52.4% [51.4, 53.4] / 51.7% [50.7, 52.6] of sampled additions are structure-confirmed.
   The confirmed half is pairs that the sequence search returns at E <= 1e-3 but that fall
   outside each query's top 60 (97.6% / 98.1% confirmed); `S-all` contains them. The half
   with no sequence hit is 17.8% / 11.9% confirmed (random pairs: 1.82%).
5. **Against every sequence arm together, context60 embeddings add pairs that mostly lack
   structural support.** Additions are 3.5% / 1.5% of accepted pairs and 13.8% / 10.6% more
   covered query genes, estimated 22% / 15% structure-confirmed. Under the predeclared
   wording: embeddings add pairs, and most of them lack structural support.
6. **The seed hybrid is where embeddings add real homology.** `S-top60` union ANN60 seeds
   keeps a clean null (0.010-0.016), adds 28.2% / 19.7% accepted pairs over `S-top60` at
   89.7% / 88.0% structure confirmation, and leaves F1 unchanged. Over the union of
   sequence arms it adds 0.7% / 0.4% of pairs, including an estimated 31,800 (650M) /
   13,800 (600M) structure-confirmed pairs with no sequence hit. These are 68.0% / 49.7%
   confirmed, against 17.8% / 11.9% for context60's no-hit additions.
7. **Weighting does not matter once the null is clean.** Unit and bit-score-percentile
   weights agree within 0.6% of blocks and 0.005 F1 excess. On context60 the percentile
   weight and raw cosine agree within 5.2% of accepted blocks; raw cosine scores up to
   0.020 higher F1.

## What was tested

- **Search.** MMseqs2 release 18 (`8cc5ce36`), `prefilter -s 7.5 --max-seqs 4000`, then
  `align -e 1e-3 --alignment-mode 3`, results read by key. 64 CPUs: prefilter 700 s, align
  846 s (search wall time 25.8 min). 63,080,078 hits for 63,013 queries; median 346 hits per
  query. 3,356 queries have all 4,000 candidates passing. For them, `S-all` is truncated at
  4,000 hits, and their 4,000th hit scores a median 0.28 of the rank-60 bit score, so the
  top-60 arms are complete.
- **Arms.** `S-all` (63.1 M anchors), `S-top60` (60 best by bit score; 3.18 M) and
  `S-top60cov50` (>= 50% coverage of both proteins; 2.92 M), each with unit or bit-score
  percentile weight. `E-ctx60` (saved context60 anchors) and `E-seed60` (saved ANN60
  seeds), each with raw cosine or cosine percentile. `H-ctx60` = `S-top60` union context60
  and `H-seed60` = `S-top60` union seeds. Embedding and hybrid arms ran for `esm2_8m`,
  `esm2_650m` and `esmc600m`.
- **One weight scale (rank-based).** A sequence anchor gets G(bits), the empirical CDF of
  bit score over `S-top60`. An embedding anchor gets F_m(cos), the empirical CDF of cosine
  over model m's ANN60 seeds. Both are top-60-per-query populations. A pair in both sources
  keeps the larger weight.
- **Chainer.** Published `chain.py` (sha256 `c7e40b8b...`) through `chain_numeric`: gap 2,
  minimum 2 anchors, paired-overlap extraction.
- **Null.** Query gene order permuted by `ordered_metadata` (seeds 17/18/19). Sequence hits
  and seeds are order-free and are re-chained on the permuted order. Context60 anchors are
  regenerated under each permutation by the frozen extension kernel with permuted query
  neighbours.
- **Replay gates (all exact).** Context60 extension equals the saved anchors pair for pair
  for all three models. `E-ctx60/cos` reproduces the published blocks (534,990 / 478,610 /
  463,807) and tonight's permuted totals (8M: 495,161 / 495,380 / 495,342). Calibrated, it
  reproduces 111,389 and 265,639 accepted blocks and F1 0.652 / 0.754 / 0.762.
- **Calibration.** Per block size 2, 3, 4, 5, 6+: the cut on mean anchor weight that holds
  permuted/observed at 5% (`cut_for`, identical to `permuted_panels.py`).
- **Scoring.** `source207` mean best-single-block F1 with 200 random placements per region
  (seed 17) and a 10,000-draw region bootstrap. Placements are identical across arms, so
  arm differences use an exact paired bootstrap.
- **Structure.** AFDB v4 models, foldseek `tmalign` with exact TM-score. Confirmed =
  max(qTM, tTM) >= 0.5. 31,495 unique sampled pairs (46,000 sample rows), first N by
  `hash(qkey*5e6+tkey)` among pairs with both structures. 11,303 of them are also in
  tonight's `tm_key.parquet`, with identical scores.

## Ablation table (5% calibration)

"Expected chance blocks" sums, over size classes, the mean permuted count above each cut.
"Covered query genes" counts genes inside the query span of any accepted block (of 67,148).

| arm | anchors | observed blocks | perm/obs | size 2 / 3 / 4 ratio | accepted blocks | accepted pairs | expected chance blocks | covered query genes | F1 (5%) | random | excess [95% CI] | F1 (all blocks) | cross-phylum (>= 10) | chain s (obs / total) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S-all/unit | 63,080,078 | 3,696,665 | 0.033 | 0.045 / 0.001 / 0.000 | 3,696,665 | 9,510,980 | 122,455 | 58,039 | 0.724 | 0.654 | +0.070 [+0.049, +0.091] | 0.724 | 2,502,534 (16,976) | 292 / 872 |
| S-all/bitrank | 63,080,078 | 3,675,859 | 0.033 | 0.046 / 0.001 / 0.000 | 3,675,859 | 9,463,514 | 122,325 | 58,062 | 0.727 | 0.654 | +0.073 [+0.052, +0.094] | 0.727 | 2,488,554 (16,975) | 319 / 996 |
| S-top60/unit | 3,176,472 | 283,549 | 0.005 | 0.008 / 0.000 / 0.000 | 283,549 | 866,264 | 1,423 | 49,277 | 0.641 | 0.586 | +0.055 [+0.021, +0.088] | 0.641 | 35,854 (27) | 22 / 44 |
| S-top60/bitrank | 3,176,472 | 283,116 | 0.005 | 0.008 / 0.000 / 0.000 | 283,116 | 865,281 | 1,423 | 49,335 | 0.647 | 0.587 | +0.060 [+0.026, +0.093] | 0.647 | 35,792 (27) | 16 / 41 |
| S-top60cov50/unit | 2,922,419 | 268,838 | 0.005 | 0.008 / 0.000 / 0.000 | 268,838 | 814,161 | 1,214 | 47,538 | 0.621 | 0.573 | +0.048 [+0.013, +0.082] | 0.621 | 33,272 (12) | 14 / 34 |
| S-top60cov50/bitrank | 2,922,419 | 268,740 | 0.005 | 0.008 / 0.000 / 0.000 | 268,740 | 813,958 | 1,215 | 47,558 | 0.623 | 0.574 | +0.050 [+0.015, +0.084] | 0.623 | 33,271 (12) | 15 / 38 |
| E-ctx60/cos@esm2_8m | 3,970,730 | 534,990 | 0.926 | 1.154 / 0.814 / 0.235 | 111,389 | 536,198 | 3,774 | 55,037 | 0.652 | 0.658 | -0.006 [-0.041, +0.028] | 0.757 | 4,474 (7) | 31 / 96 |
| E-ctx60/rank@esm2_8m | 3,970,730 | 535,028 | 0.926 | 1.153 / 0.815 / 0.235 | 112,967 | 539,041 | 3,777 | 54,800 | 0.636 | 0.657 | -0.021 [-0.058, +0.014] | 0.755 | 4,018 (7) | 23 / 91 |
| E-seed60/cos@esm2_8m | 3,970,270 | 126,775 | 0.011 | 0.016 / 0.000 / 0.000 | 126,775 | 344,086 | 1,426 | 43,421 | 0.516 | 0.535 | -0.018 [-0.054, +0.016] | 0.516 | 4,798 (0) | 13 / 45 |
| E-seed60/rank@esm2_8m | 3,970,270 | 126,775 | 0.011 | 0.016 / 0.000 / 0.000 | 126,775 | 344,083 | 1,426 | 43,419 | 0.516 | 0.535 | -0.018 [-0.054, +0.016] | 0.516 | 4,798 (0) | 14 / 48 |
| H-ctx60@esm2_8m | 6,567,031 | 829,245 | 0.974 | 1.167 / 0.973 / 0.343 | 245,957 | 1,011,738 | 9,826 | 59,754 | 0.735 | 0.709 | +0.026 [-0.001, +0.052] | 0.792 | 22,811 (40) | 45 / 161 |
| H-seed60@esm2_8m | 6,363,676 | 335,453 | 0.011 | 0.018 / 0.000 / 0.000 | 335,453 | 1,014,998 | 3,656 | 52,038 | 0.668 | 0.606 | +0.062 [+0.032, +0.091] | 0.668 | 41,531 (29) | 26 / 80 |
| E-ctx60/cos@esm2_650m | 3,971,948 | 478,610 | 0.666 | 1.051 / 0.438 / 0.139 | 265,639 | 1,052,580 | 10,133 | 64,334 | 0.754 | 0.759 | -0.005 [-0.028, +0.017] | 0.788 | 17,075 (51) | 32 / 89 |
| E-ctx60/rank@esm2_650m | 3,971,948 | 478,696 | 0.666 | 1.049 / 0.437 / 0.139 | 279,366 | 1,086,670 | 10,950 | 64,708 | 0.754 | 0.762 | -0.008 [-0.031, +0.013] | 0.787 | 17,544 (51) | 24 / 87 |
| E-seed60/cos@esm2_650m | 3,971,948 | 225,862 | 0.009 | 0.014 / 0.000 / 0.000 | 225,862 | 642,321 | 2,098 | 48,938 | 0.578 | 0.582 | -0.004 [-0.038, +0.029] | 0.578 | 11,408 (12) | 18 / 54 |
| E-seed60/rank@esm2_650m | 3,971,948 | 225,864 | 0.009 | 0.014 / 0.000 / 0.000 | 225,864 | 642,319 | 2,098 | 48,934 | 0.578 | 0.582 | -0.004 [-0.038, +0.029] | 0.578 | 11,408 (12) | 18 / 54 |
| H-ctx60@esm2_650m | 6,494,729 | 906,529 | 1.119 | 1.285 / 1.304 / 0.723 | 298,772 | 1,249,355 | 11,942 | 62,794 | 0.781 | 0.751 | +0.030 [+0.007, +0.052] | 0.819 | 28,537 (90) | 47 / 182 |
| H-seed60@esm2_650m | 5,815,345 | 367,290 | 0.010 | 0.017 / 0.000 / 0.000 | 367,290 | 1,107,529 | 3,738 | 52,966 | 0.674 | 0.612 | +0.062 [+0.032, +0.091] | 0.674 | 44,744 (44) | 26 / 76 |
| E-ctx60/cos@esmc600m | 4,027,458 | 463,807 | 0.795 | 1.110 / 0.588 / 0.155 | 147,858 | 683,194 | 5,053 | 60,888 | 0.762 | 0.715 | +0.048 [+0.023, +0.071] | 0.818 | 3,840 (13) | 29 / 88 |
| E-ctx60/rank@esmc600m | 4,027,458 | 463,893 | 0.795 | 1.109 / 0.587 / 0.155 | 146,888 | 678,435 | 4,994 | 60,201 | 0.742 | 0.710 | +0.032 [+0.004, +0.057] | 0.817 | 3,787 (13) | 22 / 81 |
| E-seed60/cos@esmc600m | 4,027,458 | 168,577 | 0.019 | 0.029 / 0.000 / 0.000 | 168,577 | 482,221 | 3,159 | 48,791 | 0.595 | 0.575 | +0.020 [-0.013, +0.052] | 0.595 | 3,744 (0) | 14 / 45 |
| E-seed60/rank@esmc600m | 4,027,458 | 168,580 | 0.019 | 0.029 / 0.000 / 0.000 | 168,580 | 482,209 | 3,161 | 48,789 | 0.595 | 0.575 | +0.021 [-0.012, +0.053] | 0.595 | 3,744 (0) | 15 / 47 |
| H-ctx60@esmc600m | 6,625,334 | 802,557 | 1.070 | 1.267 / 1.174 / 0.513 | 244,231 | 1,042,443 | 9,866 | 61,476 | 0.788 | 0.731 | +0.057 [+0.036, +0.077] | 0.830 | 20,611 (46) | 50 / 190 |
| H-seed60@esmc600m | 6,254,696 | 340,335 | 0.016 | 0.026 / 0.000 / 0.000 | 340,335 | 1,033,939 | 5,363 | 53,979 | 0.678 | 0.618 | +0.060 [+0.031, +0.089] | 0.678 | 39,164 (27) | 28 / 83 |

## Accepted blocks (5%) by target-genome stratum

Strata from GTDB r232 relative to Streptomyces (`$RD/universe/genomes.parquet`).

| arm | Streptomyces | same class | same phylum | other phylum | no GTDB |
|---|---|---|---|---|---|
| S-all/unit | 19,859 | 1,030,873 | 109,210 | 2,502,534 | 34,189 |
| S-all/bitrank | 19,480 | 1,025,025 | 108,840 | 2,488,554 | 33,960 |
| S-top60/unit | 6,048 | 232,627 | 7,854 | 35,854 | 1,166 |
| S-top60/bitrank | 5,980 | 232,323 | 7,853 | 35,792 | 1,168 |
| S-top60cov50/unit | 5,615 | 221,472 | 7,372 | 33,272 | 1,107 |
| S-top60cov50/bitrank | 5,616 | 221,372 | 7,371 | 33,271 | 1,110 |
| E-ctx60/cos@esm2_8m | 2,937 | 101,005 | 2,661 | 4,474 | 312 |
| E-ctx60/rank@esm2_8m | 3,045 | 102,928 | 2,659 | 4,018 | 317 |
| E-seed60/cos@esm2_8m | 3,693 | 114,656 | 3,280 | 4,798 | 348 |
| E-seed60/rank@esm2_8m | 3,692 | 114,657 | 3,280 | 4,798 | 348 |
| H-ctx60@esm2_8m | 4,962 | 210,527 | 6,843 | 22,811 | 814 |
| H-seed60@esm2_8m | 6,673 | 275,177 | 10,763 | 41,531 | 1,309 |
| E-ctx60/cos@esm2_650m | 5,402 | 233,800 | 8,437 | 17,075 | 925 |
| E-ctx60/rank@esm2_650m | 5,673 | 246,167 | 9,007 | 17,544 | 975 |
| E-seed60/cos@esm2_650m | 5,153 | 202,608 | 5,985 | 11,408 | 708 |
| E-seed60/rank@esm2_650m | 5,153 | 202,610 | 5,985 | 11,408 | 708 |
| H-ctx60@esm2_650m | 5,606 | 254,640 | 8,934 | 28,537 | 1,055 |
| H-seed60@esm2_650m | 7,258 | 302,158 | 11,673 | 44,744 | 1,457 |
| E-ctx60/cos@esmc600m | 3,829 | 136,920 | 2,871 | 3,840 | 398 |
| E-ctx60/rank@esmc600m | 3,727 | 136,212 | 2,761 | 3,787 | 401 |
| E-seed60/cos@esmc600m | 5,198 | 156,053 | 3,114 | 3,744 | 468 |
| E-seed60/rank@esmc600m | 5,197 | 156,057 | 3,114 | 3,744 | 468 |
| H-ctx60@esmc600m | 4,878 | 211,913 | 6,008 | 20,611 | 821 |
| H-seed60@esmc600m | 7,642 | 282,422 | 9,763 | 39,164 | 1,344 |

## Paired differences in F1 excess (arm minus reference)

Mean over 207 regions, 95% paired region-bootstrap interval (10,000 draws).

| arm | vs S-top60/bitrank | vs S-all/bitrank |
|---|---|---|
| S-all/bitrank | +0.013 [-0.010, +0.037] | - |
| S-top60/bitrank | - | -0.013 [-0.037, +0.010] |
| E-ctx60/cos@esm2_8m | -0.066 [-0.100, -0.033] | -0.079 [-0.111, -0.048] |
| E-ctx60/rank@esm2_8m | -0.081 [-0.118, -0.047] | -0.094 [-0.129, -0.060] |
| E-seed60/rank@esm2_8m | -0.078 [-0.112, -0.045] | -0.091 [-0.125, -0.058] |
| H-ctx60@esm2_8m | -0.034 [-0.054, -0.012] | -0.047 [-0.069, -0.024] |
| H-seed60@esm2_8m | +0.002 [-0.010, +0.017] | -0.011 [-0.030, +0.007] |
| E-ctx60/cos@esm2_650m | -0.065 [-0.095, -0.035] | -0.078 [-0.102, -0.054] |
| E-ctx60/rank@esm2_650m | -0.068 [-0.098, -0.037] | -0.081 [-0.105, -0.057] |
| E-seed60/rank@esm2_650m | -0.064 [-0.094, -0.036] | -0.077 [-0.108, -0.047] |
| H-ctx60@esm2_650m | -0.029 [-0.051, -0.008] | -0.042 [-0.065, -0.020] |
| H-seed60@esm2_650m | +0.002 [-0.009, +0.016] | -0.011 [-0.030, +0.007] |
| E-ctx60/cos@esmc600m | -0.012 [-0.040, +0.016] | -0.025 [-0.050, -0.002] |
| E-ctx60/rank@esmc600m | -0.028 [-0.053, -0.003] | -0.041 [-0.066, -0.017] |
| E-seed60/rank@esmc600m | -0.039 [-0.064, -0.015] | -0.052 [-0.081, -0.025] |
| H-ctx60@esmc600m | -0.003 [-0.026, +0.021] | -0.015 [-0.033, +0.002] |
| H-seed60@esmc600m | +0.001 [-0.012, +0.015] | -0.012 [-0.033, +0.007] |

## Decisive table: what each source adds (5% level, `esm2_650m` and `esmc600m`)

Reference `S-top60` = `S-top60/bitrank`. Reference `union S` = accepted pairs of all six
sequence arms (9,823,998 pairs; 58,065 covered query genes). "In search" means the
arm-only pair is an E <= 1e-3 hit of the 4,000-candidate search. "No hit" means it is not.
"Long protein" counts reference-only pairs that involve a protein over 1,022 residues
(no ESM2 vector). A block is shared when an accepted block of the other source overlaps
it on both axes in the same contig pair.

| model | arm | reference | accepted pairs arm / ref | shared | arm-only (/ ref) | arm-only: in search / no hit | ref-only (long protein) | blocks arm / ref | arm-only blocks (cross-phylum) | ref-only blocks (cross-phylum) | covered genes ref / arm / union | added by arm | added by ref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| esm2_650m | E-ctx60/rank | S-top60 | 1,086,670 / 865,281 | 534,141 | 552,529 (63.9%) | 231,154 / 321,375 | 331,140 (19,212) | 279,366 / 283,116 | 109,923 (10,203) | 109,150 (28,413) | 49,335 / 64,708 / 65,179 | 15,844 (+32.1%) | 471 |
| esm2_650m | H-ctx60 | S-top60 | 1,249,355 / 865,281 | 769,549 | 479,806 (55.5%) | 224,009 / 255,797 | 95,732 (1,705) | 298,772 / 283,116 | 67,197 (5,078) | 42,895 (12,256) | 49,335 / 62,794 / 63,168 | 13,833 (+28.0%) | 374 |
| esm2_650m | E-seed60/rank | S-top60 | 642,319 / 865,281 | 455,591 | 186,728 (21.6%) | 146,662 / 40,066 | 409,690 (19,212) | 225,864 / 283,116 | 64,305 (5,304) | 124,696 (29,715) | 49,335 / 48,934 / 52,243 | 2,908 (+5.9%) | 3,309 |
| esm2_650m | H-seed60 | S-top60 | 1,107,529 / 865,281 | 863,304 | 244,225 (28.2%) | 197,416 / 46,809 | 1,977 (59) | 367,290 / 283,116 | 86,562 (8,985) | 36 (2) | 49,335 / 52,966 / 52,971 | 3,636 (+7.4%) | 5 |
| esm2_650m | E-ctx60/cos | S-top60 | 1,052,580 / 865,281 | 526,740 | 525,840 (60.8%) | 220,204 / 305,636 | 338,541 (19,212) | 265,639 / 283,116 | 101,468 (9,925) | 114,251 (28,604) | 49,335 / 64,334 / 64,878 | 15,543 (+31.5%) | 544 |
| esm2_650m | E-ctx60/rank | union S | 1,086,670 / 9,823,998 | 747,591 | 339,079 (3.5%) | 17,704 / 321,375 | 9,076,407 (356,662) | 279,366 / - | - | - | 58,065 / 64,708 / 66,094 | 8,029 (+13.8%) | 1,386 |
| esm2_650m | H-ctx60 | union S | 1,249,355 / 9,823,998 | 972,863 | 276,492 (2.8%) | 20,695 / 255,797 | 8,851,135 (338,813) | 298,772 / - | - | - | 58,065 / 62,794 / 64,772 | 6,707 (+11.6%) | 1,978 |
| esm2_650m | E-seed60/rank | union S | 642,319 / 9,823,998 | 587,912 | 54,407 (0.6%) | 14,341 / 40,066 | 9,236,086 (356,662) | 225,864 / - | - | - | 58,065 / 48,934 / 59,571 | 1,506 (+2.6%) | 10,637 |
| esm2_650m | H-seed60 | union S | 1,107,529 / 9,823,998 | 1,041,429 | 66,100 (0.7%) | 19,291 / 46,809 | 8,782,569 (336,863) | 367,290 / - | - | - | 58,065 / 52,966 / 59,819 | 1,754 (+3.0%) | 6,853 |
| esmc600m | E-ctx60/rank | S-top60 | 678,435 / 865,281 | 413,819 | 264,616 (30.6%) | 119,428 / 145,188 | 451,462 (13,886) | 146,888 / 283,116 | 31,628 (1,741) | 163,886 (33,738) | 49,335 / 60,201 / 61,747 | 12,412 (+25.2%) | 1,546 |
| esmc600m | H-ctx60 | S-top60 | 1,042,443 / 865,281 | 713,076 | 329,367 (38.1%) | 152,548 / 176,819 | 152,205 (3,457) | 244,231 / 283,116 | 38,605 (1,774) | 69,895 (16,929) | 49,335 / 61,476 / 62,217 | 12,882 (+26.1%) | 741 |
| esmc600m | E-seed60/rank | S-top60 | 482,209 / 865,281 | 359,737 | 122,472 (14.2%) | 99,246 / 23,226 | 505,544 (14,671) | 168,580 / 283,116 | 40,485 (1,483) | 158,567 (33,539) | 49,335 / 48,789 / 52,946 | 3,611 (+7.3%) | 4,157 |
| esmc600m | H-seed60 | S-top60 | 1,033,939 / 865,281 | 863,168 | 170,771 (19.7%) | 143,008 / 27,763 | 2,113 (692) | 340,335 / 283,116 | 59,138 (3,377) | 48 (1) | 49,335 / 53,979 / 53,980 | 4,645 (+9.4%) | 1 |
| esmc600m | E-ctx60/cos | S-top60 | 683,194 / 865,281 | 409,848 | 273,346 (31.6%) | 120,994 / 152,352 | 455,433 (13,940) | 147,858 / 283,116 | 34,437 (1,740) | 165,666 (33,683) | 49,335 / 60,888 / 62,336 | 13,001 (+26.4%) | 1,448 |
| esmc600m | E-ctx60/rank | union S | 678,435 / 9,823,998 | 527,231 | 151,204 (1.5%) | 6,016 / 145,188 | 9,296,767 (348,553) | 146,888 / - | - | - | 58,065 / 60,201 / 64,199 | 6,134 (+10.6%) | 3,998 |
| esmc600m | H-ctx60 | union S | 1,042,443 / 9,823,998 | 852,312 | 190,131 (1.9%) | 13,312 / 176,819 | 8,971,686 (337,637) | 244,231 / - | - | - | 58,065 / 61,476 / 64,169 | 6,104 (+10.5%) | 2,693 |
| esmc600m | E-seed60/rank | union S | 482,209 / 9,823,998 | 451,758 | 30,451 (0.3%) | 7,225 / 23,226 | 9,372,240 (349,698) | 168,580 / - | - | - | 58,065 / 48,789 / 60,025 | 1,960 (+3.4%) | 11,236 |
| esmc600m | H-seed60 | union S | 1,033,939 / 9,823,998 | 995,408 | 38,531 (0.4%) | 10,768 / 27,763 | 8,828,590 (335,181) | 340,335 / - | - | - | 58,065 / 53,979 / 60,413 | 2,348 (+4.0%) | 6,434 |

## Structure confirmation of the additions

TM-score >= 0.5 normalized by the shorter chain; Wilson 95% intervals. Control: uniform
random query x reference pairs are 1.82% confirmed (n = 23,788; tonight's key subset).

| model | sampled set | n | confirmed | in search: n, confirmed | no hit: n, confirmed |
|---|---|---|---|---|---|
| esm2_650m | E-ctx60/rank only (vs S-top60) | 10,000 | 52.4% [51.4, 53.4] | 4,336, 97.6% | 5,664, 17.8% [16.8, 18.8] |
| esm2_650m | H-seed60 additions from embeddings | 8,000 | 89.7% [89.0, 90.4] | 5,967, 97.1% | 2,033, 68.0% [66.0, 70.0] |
| esm2_650m | shared (E-ctx60 and S-top60) | 2,500 | 96.8% [96.1, 97.5] | 2,500, 96.8% | 0 |
| esm2_650m | S-top60 only | 2,500 | 96.9% [96.1, 97.5] | 2,500, 96.9% | 0 |
| esmc600m | E-ctx60/rank only (vs S-top60) | 10,000 | 51.7% [50.7, 52.6] | 4,612, 98.1% | 5,388, 11.9% [11.1, 12.8] |
| esmc600m | H-seed60 additions from embeddings | 8,000 | 88.0% [87.2, 88.7] | 6,319, 98.2% | 1,681, 49.7% [47.3, 52.1] |
| esmc600m | shared | 2,500 | 97.8% [97.2, 98.3] | 2,500, 97.8% | 0 |
| esmc600m | S-top60 only | 2,500 | 96.2% [95.4, 96.9] | 2,500, 96.2% | 0 |

"H-seed60 additions from embeddings" are accepted hybrid pairs absent from the sequence
arm's accepted pairs and from the `S-top60` anchors. This set was added to the sample
after the hybrid block counts were seen and before any TM-score was computed (notebook,
03:52).

Additions over the union of sequence arms, structure-confirmed share estimated by
weighting the two strata above:

| arm | additions | in search / no hit | est. confirmed | est. confirmed with no hit |
|---|---|---|---|---|
| E-ctx60/rank@esm2_650m | 339,079 | 17,704 / 321,375 | 22.0% (about 74,500) | about 57,200 |
| E-ctx60/rank@esmc600m | 151,204 | 6,016 / 145,188 | 15.3% (about 23,200) | about 17,300 |
| H-seed60@esm2_650m | 66,100 | 19,291 / 46,809 | 76.5% (about 50,600) | about 31,800 |
| H-seed60@esmc600m | 38,531 | 10,768 / 27,763 | 63.3% (about 24,400) | about 13,800 |

## Verdict against the predeclared criteria

| comparison | material (>= 10% pairs or >= 5% genes) | >= 50% confirmed | predeclared reading |
|---|---|---|---|
| E-ctx60/rank vs S-top60, 650M | yes: +63.9% pairs, +32.1% genes | yes: 52.4% [51.4, 53.4] | embeddings earn their place (narrowly) |
| E-ctx60/rank vs S-top60, 600M | yes: +30.6% pairs, +25.2% genes | yes: 51.7% [50.7, 52.6] | embeddings earn their place (narrowly) |
| E-ctx60/rank vs union S, 650M | yes on genes (+13.8%); pairs +3.5% | no: est. 22% | embeddings add pairs, most lack structural support |
| E-ctx60/rank vs union S, 600M | yes on genes (+10.6%); pairs +1.5% | no: est. 15% | embeddings add pairs, most lack structural support |
| H-seed60 vs S-top60, 650M / 600M | yes: +28.2% / +19.7% pairs, +7.4% / +9.4% genes | yes: 89.7% / 88.0% | embeddings earn their place |
| H-seed60 vs union S, 650M / 600M | no: +0.7% / +0.4% pairs, +3.0% / +4.0% genes | yes: est. 76.5% / 63.3% | real but below the materiality bar |

The primary criterion passes because a 60-per-query sequence budget leaves out homologs
that the search itself finds. With exhaustive sequence anchors in the reference, the
context60 additions fall below 50% confirmation. The clean contribution is the seed
hybrid's: tens of thousands of structure-confirmed pairs with no sequence hit, which is
0.3-0.7% of the accepted pairs and +3-4% covered genes, with no benchmark gain.

## Wall time

| arm family | anchor generation | chaining (observed; 4 chains) |
|---|---|---|
| S-all | MMseqs2 search 1,546 s on 64 CPUs (prefilter 700 + align 846) | 292-319 s; 872-996 s total, 64 CPUs, 46 GB |
| S-top60, S-top60cov50 | same search, plus 29 s post-processing | 14-22 s; 34-44 s total, 24 CPUs |
| E-ctx60 / E-seed60, ESM2-650M | reference embedding 58,370 s plus query 918 s (committed embedding time, manifest); HNSW build 996 s and query 17 s on 64 workers; extension 47-72 s | 18-32 s; 54-89 s total, 24 CPUs |
| E-ctx60 / E-seed60, ESM-C 600M | embedding 11,545 + 179 s; HNSW 1,840 + 31 s; extension 64-74 s | 14-29 s; 45-88 s total |
| H-ctx60 / H-seed60 | both of the above | 26-50 s; 76-190 s total, 24 CPUs |

Embedding inference alone takes 7.5x (ESM-C 600M) to 38x (ESM2-650M) the wall time of the
whole sequence search. The manifests do not record the inference device, so this compares wall time
on different hardware.

## What works

- Pairwise sequence anchors as the base. They chain to a clean null with no context step,
  need no learned weight, and give the best F1 excess. Exhaustive hits (`S-all`) give
  2.49 M cross-phylum blocks at 3.3% chance.
- ANN60 seeds as an addition to sequence anchors (`H-seed60`). The null stays clean, the
  additions are 88-90% structure-confirmed, and half to two-thirds of the additions with no
  sequence hit are confirmed.
- The replay discipline. Every embedding arm reproduced its published blocks and
  tonight's permuted totals exactly before anything new was computed.

## What does not

- Context60 neighbourhood selection as an anchor source. In every form (alone, with
  cosine or percentile weight, or merged with sequence anchors) it brings chance to
  0.67-1.12 permuted/observed. The 5% cut then rejects most small blocks, including
  sequence-supported ones: `H-ctx60` keeps 20,611-28,537 cross-phylum blocks against
  35,792 for `S-top60` alone. Its additions with no sequence hit are 12-18% confirmed.
- Embedding anchors alone on the benchmark. Every embedding-only arm has lower F1 excess
  than `S-all` (by 0.025 to 0.094, all intervals below 0).
- Weighting as a calibration lever for sequence anchors. Unit and bit-score weights are
  indistinguishable because all sizes already pass.

## Artifacts

Local (`benchmarks/redesign/ws_seqanchors/`):
- `code/`: `run_search.sbatch`, `search_post.py`, `sa.py` (arms, null, calibration, metrics),
  `run_emb.sbatch`, `run_stop60.sbatch`, `run_sall.sbatch`, `run_hyb.sbatch`, `compare.py`
  (arm tables, decisive counts, structure sample, TM summary, paired bootstrap),
  `run_compare.sbatch`, `run_compare_smoke.sbatch`, `run_paired.sbatch`. Identical by sha256
  to the remote copies.
- `results/`: `arms.tsv`, `arms_by_stratum.tsv`, `decisive_counts.tsv`,
  `structure_confirmation.tsv`, `paired_f1_excess.tsv`, `per_region_excess.tsv`,
  `search_stats.json`, `tm_prep_stats.json`, `RESULTS.md` (auto-generated), `arms/<arm>.json`
  (per-size calibration, strata, per-region F1, timing), `ctx_perm/<model>.json` (replay
  records).

Remote (`$RD/ws_seqanchors/`):
- `work/hits.parquet` (63.1 M hits: rows, keys, bits, identity, E-value, coverage, ranks).
- `work/arms/<arm>/` (`observed_blocks.parquet`, `permuted_blocks.parquet`,
  `block_calls.parquet`, `pairs.parquet`, `covered_query_rows_{05,01}.npy`).
- `work/ctx_perm/<model>_seed<s>.parquet` (context60 regenerated under each permutation).
- `work/tm/` (`sample_scored.parquet`, `aln_tm.tsv`).
- `logs/`. Jobs: 76256 search, 76259 embedding arms, 76280 / 76281 / 76282 sequence and
  hybrid arms, 76283 comparison and TM-align, 76355 smoke test, 76377 paired bootstrap.

## Open questions

1. A profile-level sequence search (MMseqs2 profiles or HHblits against family profiles)
   may recover the seed hybrid's 13,800-31,800 confirmed pairs with no hit. That would
   settle whether any embedding contribution survives a stronger sequence baseline.
2. The `S-all` size-2 class sits at 4.5% chance, inside the 5% rule. A per-block weight with
   discriminating power, such as a calibrated probability of homology or the
   co-localization statistics of the engine workstream, could sharpen it. Unit and bit
   weights carry none.
3. The 207-region benchmark sits in actinomycete genomes with close references. A
   divergent collection (the DPANN package) is where sequence-invisible homology should
   matter most. The seed hybrid is the arm to carry there.
4. The calibrated-probability weight scale (P(TM >= 0.5 | score) per source) was left
   untested. The rank scale was adequate here because the clean-null arms accept
   everything.
5. For the 3,356 queries that saturate the cap, `S-all` holds only their best 4,000 hits.
   Raising `--max-seqs` would add paralog-family anchors and chaining cost.
