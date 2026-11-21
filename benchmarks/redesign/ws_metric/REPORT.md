# ws_metric report

Copied verbatim by the coordinator from the closing entries of `ws_metric/NOTEBOOK.md` (line 653 to the end); the agent's own REPORT.md write was blocked by the harness.

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
