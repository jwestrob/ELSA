# ELSA redesign: shared brief for all workstreams

Written 11 November 2026 by the coordinating session. Read all of it before you
touch anything. It records what was established tonight, where every input
lives, the traps already hit, the cluster rules, and how to record your work.

Jacob West-Roberts (the user) authorized this program: try the proposed changes,
ablate them properly, and record everything thoroughly. He wants an architecture
that is new, in place of a pairwise-collinearity design with embedding anchors.

## 1. What ELSA is today

ELSA finds conserved gene neighbourhoods from protein-language-model embeddings.

1. One mean-pooled, L2-normalized vector per protein.
2. Top-k cosine neighbours per protein in other genomes (k + 10 = 60), cosine >= 0.85.
3. Production only: "neighbourhood selection" (`elsa/neighborhood.py`). For each query
   protein it proposes targets that sit next to the targets of adjacent query genes,
   then keeps 60 targets ranked by supported-first, then witness + focal cosine.
4. Per contig pair, dynamic-programming chains of collinear anchors, gap <= 2 genes on
   both axes, longest chain first and summed cosine second (`elsa/chain.py`).
5. Greedy extraction of blocks, then clustering of pairwise blocks by interval overlap.

Two checkouts exist locally and have diverged:
`~/Documents/Sandbox/elsa_test/ELSA` (research; branch `agent/normalized-result-contract`)
and `~/Documents/Sandbox/elsa_test/ELSA-production` (branch `feat/neighborhood-chaining`,
commit `e7810bb`, the benchmarked chainer). Do not edit either `elsa/` package. Copy
what you need into your workstream directory.

## 2. What was measured tonight (Spacedust bacterial collection)

Collection: 9 actinomycete query genomes (67,148 proteins) against 1,308 reference
genomes (4,190,383 proteins). The reference holds 1 Streptomyces genome, 132 other
genomes of class Actinomycetes, 25 other Actinomycetota, 1,127 genomes of other
phyla and 23 without a GTDB match. The benchmark is 207 annotated BGC regions
(`source207`), scored as mean best-single-block F1 on query gene intervals, with
200 random placements of each annotation as a control.

Saved runs exist for seven models (ESM2 8M/35M/150M/650M, ESM-C 300M/600M/6B), each
with seeds, retained anchors and blocks ("context60" = step 3 on).

**A. Half of chained anchor pairs have no pairwise sequence alignment.** Smith-Waterman
(MMseqs2) E > 1e-3 against the full reference for 46-48% of ESM2 chained pairs and
52-58% of ESM-C pairs. Native Spacedust pairs: 9.3%. Random pairs: 99.9%.

**B. Small blocks are at chance level.** With gene order permuted before step 3, the
ratio of permuted to observed blocks is 1.15, 0.81, 0.23, 0.023 and 0.0002 for blocks
of 2, 3, 4, 5 and 6+ anchors (ESM2-8M; ESM-C 6B within 0.1). Blocks of 2-3 anchors
are 84% of all blocks.

**C. Step 3 creates the chance blocks.** Plain global top-60 without step 3 gives
116,931 blocks with a permuted/observed ratio of 0.012 and 97% sequence-visible pairs.
Step 3 gives 534,990 blocks at ratio 0.93. Cause: the "supported" gate is binary and
is satisfied by any cosine >= 0.85, which nearly every protein pair meets in raw ESM
space.

**D. A calibrated acceptance rule works.** For each block size, the cut on mean anchor
cosine that holds permuted/observed at 5% keeps 111,389 blocks (ESM2-8M) to 265,639
(ESM2-650M). Accepted blocks are 72-84% sequence-visible by pair; rejected small
blocks are 11-37%.

**E. Rescored benchmark (blocks calibrated to 5%).** F1: ESM2 8M 0.652, 35M 0.693,
150M 0.749, 650M 0.754; ESM-C 300M 0.744, 600M 0.762, 6B 0.749; native Spacedust
0.676. On all blocks the same models score 0.757-0.819, inflated by chance blocks.
F1 minus random placement: about 0 for every ESM2 model, +0.040 to +0.081 for ESM-C,
+0.065 for Spacedust.

**F. The global top-60 budget cannot reach distant genomes.** 87-97% of chained pairs
target Actinomycetota genomes (14% of reference proteins). Exhaustive top-5 per
reference scaffold (ESM2-8M, 519 M candidates) gives 186,529 calibrated cross-phylum
blocks against 4,474, and 12,233 cross-phylum blocks of 10+ anchors against 7.

**G. Lineage signal in the embeddings.** ESM-C 6B sends 8.9% of its chained pairs to
the single Streptomyces genome and 77% of those are sequence-invisible. It also makes
the most chance blocks.

Pending when this was written: TM-align scores for the sequence-invisible pairs
(jobs 76209 and 76134) and a first comparison of pair-scoring functions (job 76221).
Check section 4 paths for whether they have landed.

## 3. The program

The target architecture is **recurrence-first**: assign every protein a discrete
family token, find token pairs that sit near each other in many genomes, test that
recurrence against a within-genome shuffle, and read loci and modules off the
resulting support landscape. Cost is linear in genes. There is no per-gene
neighbour budget and no pairwise significance step.

| Workstream | Directory | Owner | Scope |
|---|---|---|---|
| engine | `engine/` | coordinator | co-localization statistics, nulls, loci, benchmark scoring |
| tokens | `ws_tokens/` | agent | family-token arms and token quality |
| metric | `ws_metric/` | agent | similarity metric learned from synteny; centring; local scaling |
| chain | `ws_chain/` | agent | evidence-weighted pairwise chaining with a built-in null |
| seqanchors | `ws_seqanchors/` | agent | sequence anchors and hybrid anchors through the same chainer |
| dpann | `ws_dpann/` | agent | DPANN benchmark package with curated-system gold loci |
| priorart | `ws_priorart/` | agent | closest prior work and an honest novelty assessment |

Each workstream owns exactly one local directory,
`~/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/<workstream>/`, and one remote
directory, `$RD/<workstream>/`. Write nowhere else, with one exception: finished
token files go to `$RD/tokens/` (section 5).

## 4. Data map (biotite, `ssh jwestrob@biotite.berkeley.edu`)

```
ROOT=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
S=$ROOT/seq_invisible_20261111        # tonight's pair-level experiment
SCALE=$ROOT/model_scaling_20260913    # seven-model runs
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
```

`$PY` has duckdb 1.5.1, pyarrow, pandas 2.3.3, numpy 2.2.6, torch 2.10.0 (CPU works),
scikit-learn 1.7.2, faiss-cpu, hnswlib and numba.

**Gene universe (use this for every join).** `$RD/universe/genes.parquet`, 4,257,531
rows, columns `gid, side, row_index, mmseqs_key, gene_id, genome, genome_code, contig,
contig_code, position_index, strand, length, has_struct`.
- `gid` = ELSA reference `row_index` for the 4,190,383 reference proteins, and
  4,190,383 + ELSA query `row_index` for the 67,148 query proteins.
- `mmseqs_key` is the key in the MMseqs databases below (a different numbering).
- `position_index` is the gene's rank on its contig; `contig_code` is global.
- `$RD/universe/genomes.parquet`: one row per genome with `genome_code, side, n_genes,
  phylum, class, ord, family, genus, species, stratum`. Query genomes have no GTDB row.
- `$RD/universe/contigs.parquet`: `genome, contig, contig_code, n_genes`.

**Embeddings.** For model `m` in `esm2_8m, esm2_35m, esm2_150m, esm2_650m, esmc300m,
esmc600m, esmc6b`, read `vectors.query.path` and `vectors.reference.path` from
`$SCALE/results/<m>/analysis/full_workflow/protocol.json`. They are float32 `.npy`
matrices, L2-normalized, row = ELSA `row_index`; open with `np.load(..., mmap_mode="r")`.
A row of NaN means no vector (ESM2 models skip proteins over 1,022 residues: 936
query and, for 35M-650M, 75,331 reference proteins). Dimensions: 320, 480, 640, 1280,
960, 1152, 2560. The 6B reference matrix is 43 GB.

**Saved pairwise runs.** `$SCALE/results/<m>/analysis/full_workflow/`:
- `seeds.parquet`: `query_row_index, target_row_index, similarity` (original ANN60).
- `anchors.parquet`: retained context60 anchors, plus `is_original, witness_*, context_score`.
- `all_blocks.parquet`: `block_id, query_genome, target_genome, query_contig,
  target_contig, query_start, query_end, target_start, target_end, n_anchors,
  chain_score, orientation, anchor_query_ids, anchor_target_ids, anchor_query_gene_ids,
  anchor_target_gene_ids` (the four anchor columns are JSON lists).
- Pipeline code: `$SCALE/code/run_spacedust_scaling.py` (`complete_metadata`,
  `chain_numeric`, `neighbors`, `load_published`), `evaluate_spacedust_queries.py`
  (`evaluate`), `spacedust_prepared_extension_executor.py` (`extend_prepared_queries`).
  The published chain snapshot is `$SCALE/production/elsa/`.
- Other retrieval arms for ESM2-8M, each with `observed` and `order_seed17/18/19`:
  `$ROOT/full_reference_interventions_20260910/results/{scaffold_k5,global60}/alternatives/`.

**Benchmark.** `$ROOT/evaluation/deepbgc_s8_207/regions.parquet` (207 regions).
Query metadata `$ROOT/bgc_queries/metadata.parquet`; reference metadata
`$ROOT/results/esm2_t6/shard_000/export/metadata.parquet`.

**Sequences and structures.** MMseqs databases `$ROOT/native_run_20260910/query/querydb`
and `.../reference/keggclusterdb`. Foldseek databases with AlphaFold DB v4 models
`.../query/querydb_foldseek` (65,882 of 67,148 proteins) and
`.../reference/keggclusterdb_foldseek` (3,809,014 of 4,190,383). FASTA exports:
`$ROOT/bgc_queries/proteins.faa` and `$ROOT/results/esm2_t6/shard_000/export/proteins.faa`
(verify the header format before relying on it). Native Spacedust output:
`$ROOT/native_run_20260910/results/all_blocks.parquet` (1,641,181 cluster hits).

**Tonight's pair-level tables (`$S/work/`).**
- `aln_seq.tsv`: 12,676,988 pairs, tab-separated, no header: `qkey, tkey, bits, seqid,
  evalue, qstart, qend, qlen, tstart, tend, tlen` (keys are `mmseqs_key`). Every pair
  returned an alignment at `-e inf`. "Sequence-visible" means `evalue <= 1e-3`.
- `membership.parquet`: `model, cls, qkey, tkey, similarity, is_original, n_anchors_max,
  n_blocks`, with `cls` in `chained, anchor_unchained, seed_unchained, spacedust, random`.
- `pairs.parquet`, `sd_pairs.parquet`, `qx.parquet`, `tx.parquet`.
- `pair_table.parquet`: sequence plus TM-scores for all pairs. Exists once job 76134
  finishes (several hours); check before use.
- `$S/work_key/tm_key.parquet` (`qkey, tkey, tm_max, tm_min, lddt`) and
  `block_pairs.parquet`: TM-scores for sequence-invisible pairs inside calibrated
  blocks, from job 76209. Check before use.
- `$S/results/permuted_panels/<m>.json`, `<m>_block_calls.parquet` (`block_id, n_anchors,
  m, n_vis, accept05, accept01`), `<m>_permuted_blocks.parquet`.
- Reference implementations: `$S/code/pipeline.sh` (pairwise MMseqs and TM-align on an
  explicit pair list through a fake prefilter database), `permuted_panels.py` (valid
  permutation, calibration, benchmark scoring), `key_subset.py`, `anchor_evidence.py`.
- Local mirror of small results:
  `benchmarks/evaluation/spacedust_2026-09-09/seq_invisible_20261111/`.

## 5. Interfaces between workstreams

**Token files.** One arm = one file `$RD/tokens/<arm>.parquet` with columns `gid BIGINT,
token BIGINT`, exactly 4,257,531 rows, `token = -1` for a protein without a token.
Beside it, `<arm>.json` with every parameter, input path, row count, token count,
wall time and the code hash. Arm names: `<source>__<model>__<transform>__<detail>`,
for example `kmeans__esm2_650m__raw__K262144` or `mmseqs__seq__id30c80`. Append one
line per finished arm to `$RD/tokens/INDEX.tsv` (`arm, n_tokens, n_assigned, seconds,
date`). Write to a temporary name and rename, so a reader never sees a partial file.

**Transformed vectors.** A learned or centred representation that another workstream
should tokenize goes to `$RD/ws_metric/vectors/<name>.npy` in `gid` order (float32),
with a `.json` manifest.

**Block and locus tables.** Anything scored on `source207` needs `query_genome,
query_contig, query_start, query_end` in gene-index coordinates. Use `evaluate` from
`$SCALE/code/evaluate_spacedust_queries.py` with `controls=200, seed=17`.

## 6. Traps already hit. Do not repeat them.

1. **Row index and key are different numberings.** ELSA `target_row_index` is its own
   order; the MMseqs key is `source_sequence_db_key` (also the number inside `gene_id`,
   `spacedust-kegg70:<key>`). Use the universe table.
2. **Anchor lists are stored sorted per side.** In a block with `orientation = -1`, the
   target list pairs back to front: reverse it before zipping with the query list.
   Assert that every pair you derive exists in `anchors.parquet`.
3. **A permutation null must permute before every order-dependent step.** Re-chaining
   anchors that were selected with the true gene order gives 0.5-1.4% chance blocks
   where the valid null gives 93%. Use `ordered_metadata` from `$S/code/permuted_panels.py`
   (seeds 17, 18, 19) and replay the saved totals first: ESM2-8M 495,161 / 495,380 /
   495,342 and ESM-C 6B 535,642 / 532,999 / 534,641.
4. **Raw cosine 0.85 is not a threshold.** Unrelated proteins sit near 0.93 in ESM2-8M.
5. **All-blocks F1 is inflated.** Report F1 on calibrated blocks, with the random
   placement control, and the permuted/observed ratio beside every block count.
6. **Reference MMseqs headers are not names.** Read results by key (`prefixid --tsv`),
   or alias the database with a header database of `K<key>` as `$S/code/pipeline.sh` does.
7. **TM-align costs about 0.1 s per pair** and seconds for multi-thousand-residue
   proteins. Sample before you align structures.
8. **Chain scores differ by up to 1e-4 between AVX2 and AVX512 nodes.** Compare block
   paths and counts exactly, scores with that tolerance.
9. **Structural similarity means shared fold, not orthology.** Keep the two apart.
10. **Never run `find`, `du` or `grep -r` over `/home/jwestrob`, `/groups/banfield` or
    any large tree.** Use `ls` on known paths.

## 7. Cluster rules

Before your first `sbatch`, read both files in full, in the same working turn:
`/home/jwestrob/.claude/CLAUDE.md` and
`/home/jwestrob/jwestrob/obsidian-vault/biotite-cluster-ops.md`. Repeat after any
interruption. The essentials:

```
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J <name>
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o <absolute log path>_%j.log
```

- No `--mem`, no `--time`, no `-N`, no subprocess timeouts. 24 CPUs come with 96 GB.
- A job that truly scales to 64 threads or needs up to 256 GB: `--qos=standard_expand_2
  -c 64`, at most **one such job per workstream at a time** (the account allows 10
  submitted in total).
- Every parallel stage uses `$SLURM_CPUS_ON_NODE` threads. A single-threaded job never
  goes to Slurm: parallelize it or run it inline if it is small.
- **CPU only.** The H200 node is drained. Request no GPU.
- **Job budget on `standard_expand`** (20 running per account, shared by all of us):
  tokens 5, chain 4, metric 3, seqanchors 3, dpann 2, coordinator 3. Stay inside yours.
- **Login node: at most 2 threads per workstream**, and only for work under about ten
  seconds and 1 GB. Anything heavier is a Slurm job.
- Inspect the wrapper before `sbatch`, check the job with `scontrol show job` after,
  and poll `sacct` until a terminal state. Chain dependent jobs with
  `--dependency=afterok:<id>`.
- All compute runs on biotite. Use the Mac only to edit files and to `scp`/`ssh`.

## 8. Recording rules

Keep `NOTEBOOK.md` in your local workstream directory, append-only, with a timestamp
on every entry. Record, as you go:
- the question and the predeclared criterion **before** each experiment (what result
  would count for or against the idea);
- exact commands, job IDs, input paths and code file names;
- result tables with exact counts, and intervals where a comparison is made;
- negative results stated plainly, and every correction to an earlier entry.

Finish with `REPORT.md`: what was tested, the ablation table, what works, what does
not, artifact paths, open questions. Mirror code and small result files (JSON, TSV,
Markdown) to the local directory; leave large files on biotite. Do not run git
commands that change the repository; the coordinator commits.

## 9. Rigor

- Hold out data for anything trained or tuned: query genomes, and target phyla where
  the claim is about distant homology.
- Every comparison holds everything else fixed and names what differs.
- Every enrichment or yield number needs its null or control beside it.
- Numbers, not adjectives. Report a negative result as a result.
- Biology: report observed domains or hits as observations. A system, family or
  pathway name is allowed only when a curated caller emitted that exact call; check
  which curated resources exist rather than assuming. Otherwise tag UNVERIFIED.
- Writing: lead with the finding in direct affirmative language, define terms on
  first use, and keep qualifications to where they change the interpretation.
