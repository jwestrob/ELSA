# ws_tokens report

Copied verbatim by the coordinator from the closing entries of `ws_tokens/NOTEBOOK.md` (line 577 to the end); the agent's own REPORT.md write was blocked by the harness.

## 2026-11-11 07:25 MST: final summary, ablation table and recommendation

(The harness refuses a REPORT.md from this subagent session; this entry is the report. All
TM columns below are the re-scored values of job 76460, own-sample only.)

### What was tested

52 token arms over all 4,257,531 genes, each in `$RD/tokens/<arm>.parquet` + `.json`, one
line each in `$RD/tokens/INDEX.tsv`:
- `mmseqs__seq__id30c80` (sequence families, the baseline and reference labelling);
- 30 k-means arms: ESM2-8M {raw, globalcentred, genomecentred}, ESM2-650M / ESM-C 600M /
  ESM-C 6B {raw, genomecentred}, each at K = 65,536 / 262,144 / 1,048,576;
- 7 graph arms: ESM2-8M genome-centred mutual 20-NN graph, Leiden modularity at resolution
  1 to 30,000 (ESM2-650M graph: job 76395 still running at this entry; its quality job 76462
  and a final `run_finalize` 76463 are queued with dependencies and will add 4 rows to
  `quality.tsv`);
- 17 hybrid arms: MMseqs families merged by ESM2-650M genome-centred centroid similarity,
  union-find (3, percolate) or Leiden CPM (14, sweep of the merge stringency).

### Ablation table (representative rows; every arm is in `quality.tsv`)

Recall columns: share of labelled pairs sharing a token. Precision: same-token
cross-genome pairs, 200,000 per sampler; "hom" = sequence-visible or TM >= 0.5.

| arm | tokens | sd_vis | sd_invis | tm_hi | tm_lo | random | seq prec (pair) | hom prec (pair) | hom prec (token) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **mmseqs id30c80** | 1,038,303 | 0.555 | 0.005 | 0.011 | 0.0004 | 6e-5 | 0.994 | 0.9997 | 0.997 |
| kmeans 8m raw K262144 (arm 1) | 262,158 | 0.040 | 0.003 | 0.030 | 0.003 | 1e-5 | 0.490 | 0.637 | |
| kmeans 8m gnm K65536 | 65,539 | 0.094 | 0.010 | 0.062 | 0.004 | 1e-5 | 0.482 | 0.659 | |
| kmeans 8m gnm K262144 | 262,158 | 0.044 | 0.003 | 0.027 | 0.002 | 0 | 0.537 | 0.681 | 0.645 |
| kmeans 650m gnm K65536 | 65,494 | 0.159 | 0.014 | 0.149 | 0.012 | 0 | 0.582 | 0.712 | 0.753 |
| kmeans 650m gnm K1048576 | 1,048,563 | 0.020 | 0.001 | 0.016 | 0.002 | 0 | 0.694 | 0.772 | 0.850 |
| kmeans esmc600m gnm K65536 | 65,536 | 0.084 | 0.003 | 0.053 | 0.006 | 3e-5 | 0.276 | 0.371 | |
| kmeans esmc6b raw K65536 | 65,547 | 0.049 | 0.002 | 0.054 | 0.007 | 1e-5 | 0.165 | 0.247 | |
| kmeans esmc6b gnm K1048576 | 1,048,555 | 0.012 | 0.000 | 0.010 | 0.001 | 0 | 0.401 | 0.490 | |
| graph 8m gnm r100 | 197,873 | 0.556 | 0.264 | 0.372 | 0.026 | 3e-4 | 0.337 | 0.586 | 0.703 |
| graph 8m gnm r3000 | 247,268 | 0.169 | 0.021 | 0.139 | 0.010 | 2e-5 | 0.627 | 0.790 | 0.642 |
| hybrid union-find p=1e-5 | 956,213 | 0.638 | 0.009 | 0.030 | 0.007 | 6e-4 | 0.163 | 0.223 | |
| **hybrid cpm p=1e-4** | 956,330 | 0.619 | 0.008 | 0.025 | 0.001 | 1e-4 | 0.991 | 0.9994 | 0.852 |
| hybrid cpm p=1e-3 | 874,916 | 0.640 | 0.012 | 0.044 | 0.002 | 1e-4 | 0.987 | 0.9992 | 0.772 |
| hybrid cpm p=1e-1 | 608,932 | 0.704 | 0.044 | 0.093 | 0.004 | 1e-4 | 0.945 | 0.9976 | 0.680 |
| **hybrid cpm r=0.05** | 334,660 | 0.799 | 0.142 | 0.259 | 0.012 | 2e-4 | 0.811 | 0.966 | 0.601 |
| hybrid cpm r=0.002 | 254,027 | 0.898 | 0.417 | 0.472 | 0.028 | 4e-4 | 0.542 | 0.848 | 0.640 |

n: sd_vis 3,955,569; sd_invis 403,491; tm_hi 72,372; tm_lo 497,092; random 300,000.

Recall on sequence-visible Spacedust pairs by target stratum (other phylum, n = 3,142,700):
MMseqs 0.450, hybrid cpm p=1e-4 0.508, p=1e-1 0.595, r=0.05 0.698, r=0.002 0.830,
graph r100 0.447, kmeans 650m gnm K65536 0.029. By identity < 30% (n = 745,389): 0.130,
0.168, 0.255, 0.447, 0.734, 0.353, 0.018.

### Predeclared criteria

- F1 (pair sequence precision < 50% at every K): holds for ESM-C 600M and 6B (all arms
  < 0.48), not for ESM2 (0.56-0.69 at K1048576).
- **F2 (sd_vis recall < 50% at every K): holds for every k-means arm (max 0.159).** k-means
  embedding tokens are a poor family stand-in on this collection.
- F3 (collision > 0.5%): holds for no arm except graph r1 (2.2%).
- B1-B4 (embedding tokens better than sequence clusters): not met by any k-means or graph
  arm (homology precision <= 0.79 against 0.9997).
- Hybrid criterion (tm_hi +0.05 over MMseqs, seq precision >= 0.9, homology >= 0.95): met by
  CPM p = 1e-2, 3e-2, 1e-1 (tm_hi 0.075-0.093); r = 0.1 and r = 0.05 meet the recall and
  homology parts with sequence precision 0.85 / 0.81.
- Recommendation rule (highest tm_hi among arms with homology precision >= 0.80 and
  collision <= 0.1%): the gate never binds inside the CPM sweep, so the literal rule picks
  the loosest arm tested (r = 0.002). The rule was written for discrete arms and has no
  stopping point on a continuous stringency parameter; I do not use its literal output.

### Recommendation for the engine

1. **Primary: `mmseqs__seq__id30c80`.** 99.97% pair and 99.65% per-token homology precision,
   collision 6e-5. Its weakness is family completeness: 55.5% of visible Spacedust pairs, 45%
   of cross-phylum ones, 13% below 30% identity.
2. **Second arm for distant recurrence: `hybrid__esm2_650m__genomecentred__mm30cpmr0p05`.**
   Visible-pair recall 0.799 (cross-phylum 0.698), invisible 0.142, tm_hi 0.259 with tm_lo
   0.012 (fold discrimination 21x), pair homology precision 0.966, collision 2e-4. Its cost
   is per-token purity: 60% homology precision for a uniformly chosen token, against 99.7%
   for MMseqs. Recurrence calls found only under this arm should be checked for token purity
   (per-token TM) before they are believed.
3. **Conservative alternative: `...mm30cpm1em4`** (per-token homology 0.85, pair 0.9994):
   +6.4 points of visible recall over MMseqs, little remote homology.
4. Do not use k-means arms (including arm 1) as family tokens; they serve only as a
   negative control. If the engine needs an embedding-only tokenization for an ablation, use
   graph r100 (recall equal to MMseqs on visible pairs, homology precision 0.59).

### Costs (wall time, CPUs, peak RSS)

MMseqs clustering 692 s on 64 CPUs, 6.5 GB. k-means: ESM2-8M 147-181 s, ESM2-650M 513-735 s,
ESM-C 600M 451-611 s on 24 CPUs (10-41 GB); ESM-C 6B 436-894 s on 64 CPUs (81 GB); the
coarse codebook and data load are 80-90% of each. Graph ESM2-8M: 3,948 s IVF kNN (memory-
bandwidth bound) + 1-3 min per Leiden resolution, 18 GB. Hybrid: 1,867 s once (centroids +
family kNN), then 10-50 s per merge setting from the cache, 41 GB. Pair precision: MMseqs
SW of ~4 M pairs in 1.5 min; TM-align about 0.05 s per pair per core once chunked.

### Artifacts

- Tokens: `$RD/tokens/*.parquet`, `*.json`, `INDEX.tsv` (52 arms of this workstream; the
  `dev__esm2_8m__raw__K16384` files there are not from this workstream).
- Quality: `$RD/ws_tokens/quality.tsv` (one row per arm), `$RD/ws_tokens/recall_strata.tsv`,
  per-arm `$RD/ws_tokens/quality/<arm>/{stats.json, recall.tsv, sample.parquet,
  sample_aln.parquet, tm_invisible*.parquet}`; labels `$RD/ws_tokens/labels/`.
- Reusable intermediates: gid-keyed MMseqs DB `$RD/ws_tokens/work/mmseqs/seqdb` (+ FASTA
  `all.faa`, headers = gid); key-headed foldseek aliases `$RD/ws_tokens/work/fsdb/`; coarse
  codebooks `$RD/ws_tokens/work/kmeans/<model>__<transform>/`; ESM2-8M genome-centred 20-NN
  lists `$RD/ws_tokens/work/graph/esm2_8m__genomecentred/knn_k20_np32.npz`; hybrid family
  graph `$RD/ws_tokens/work/hybrid/esm2_650m__genomecentred/edges_k10_np32_*.npz`.
- Code: `$RD/ws_tokens/code/` (mirror `code/` here), snapshots by hash in
  `$RD/ws_tokens/code_snapshots/`; small results mirrored in `results/`.

### Open questions

- Per-token purity of the CPM hybrids: is the 60% driven by a few merge-prone fold classes
  (repeats, transporters), so that a per-token purity filter could rescue most merges?
- Whether a learned metric (ws_metric) or ESM2-650M graph communities (job 76395) lift the
  graph arm above the 0.79 homology-precision plateau of ESM2-8M.
- The tm_hi / tm_lo / blk650 label sets were chosen as ESM anchor pairs and favour arms
  built on ESM cosine; sd_invis is the unbiased check and agrees in direction.

## 2026-11-11 08:16 MST: hand-off state

Still running at hand-off: 76395 (ESM2-650M genome-centred graph, 64 CPUs; IVF 20-NN search
at 2 h 17 min, 135 CPU-hours so far, against 26 CPU-hours for ESM2-8M). Queued behind it:
76462 (quality with TM for `graph__esm2_650m__genomecentred__mknn20r{100,300,1000,3000}`,
afterok 76395) and 76463 (`run_finalize.sbatch`: recall for every arm, `quality.tsv`,
`recall_strata.tsv`; afterany 76462). Their rows will appear in `$RD/ws_tokens/quality.tsv`
without further action; the local mirror in `results/` predates them.
Local `code/` matches remote `$RD/ws_tokens/code/` byte for byte (sha256, 07:20 MST).

## 2026-11-11 09:05 MST: ESM2-650M graph arm (closes the hand-off item above)

Job 76395 finished in 2 h 56 min on 64 CPUs (IVF 20-NN search 10,217 s, peak RSS 50 GB):
4,181,264 nodes, 21,240,201 mutual edges, 156,551 isolated, mean degree 10.16. Quality 76462.

| graph 650m gnm | tokens | max | sd_vis | sd_invis | blk650_vis | tm_hi | tm_lo | random | seqprec pair | TMconf | homology prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r100 | 195,594 | 8,497 | 0.390 | 0.146 | 0.910 | 0.495 | 0.038 | 1.5e-4 | 0.408 | 0.368 | 0.626 |
| r300 | 202,928 | 3,916 | 0.292 | 0.074 | 0.893 | 0.444 | 0.032 | 6e-5 | 0.526 | 0.421 | 0.726 |
| r1000 | 216,409 | 865 | 0.237 | 0.041 | 0.858 | 0.356 | 0.028 | 2e-5 | 0.639 | 0.483 | **0.813** |
| r3000 | 238,208 | 372 | 0.197 | 0.024 | 0.797 | 0.279 | 0.022 | 2e-5 | 0.710 | 0.508 | **0.857** |

- The larger model lifts the graph arm past the ESM2-8M plateau (0.79): r1000 and r3000 clear
  the 0.80 homology gate, the only embedding-only arms that do.
- Its gains sit on the ESM-selected label sets (tm_hi 0.28-0.36, blk650_vis 0.80-0.86); on
  the unbiased Spacedust sets it trails the CPM hybrid r = 0.05 (sd_vis 0.197-0.237 vs 0.799,
  sd_invis 0.024-0.041 vs 0.142) at lower homology precision (0.81-0.86 vs 0.966).
  The recommendation stands.
- Under the predeclared rule restricted to embedding-only arms, the pick is
  `graph__esm2_650m__genomecentred__mknn20r1000` (tm_hi 0.356 at homology precision 0.813).

## 2026-11-11 09:15 MST: final state

`run_finalize` 76463 completed: `$RD/ws_tokens/quality.tsv` has 56 rows (all arms of this
workstream), `recall_strata.tsv` covers every arm; both mirrored to `results/` with all
manifests and per-arm `stats.json` / `recall.tsv`. No jobs of this workstream remain queued.
