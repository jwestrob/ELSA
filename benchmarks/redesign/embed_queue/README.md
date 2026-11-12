# embed_queue

A resumable protein-embedding queue served by independent Slurm workers.

- Biotite root: `/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/`
  (`code/`, `runs/<run>/`, `logs/`, `hf_home/`).
- Repo copy of the code: `benchmarks/redesign/embed_queue/`.
- First run: `runs/dpann__esmpp_small__20261111` (DPANN, 3,585,303 proteins, ESM++ small).

## Design

`prep` turns a FASTA into a run directory: sequences are normalized (uppercase, terminal `*`
removed, every symbol outside `ACDEFGHIKLMNPQRSTVWYXBUZO` mapped to `X`), exact duplicates
collapse to one unique sequence, uniques are sorted by length and cut into shards of about
4,000,000 tokens. Each shard gets a token file in `queue/pending/`.

A worker is one Slurm job on one GPU. It loops:

1. claim a shard by renaming `queue/pending/<shard>` to `queue/claimed/<shard>@<job>_<host>_<pid>`
   (atomic, so two workers never hold the same shard);
2. embed it in length-sorted batches bounded by padded tokens and by B x T^2;
3. write `out/<shard>.npy` through a temporary file and a rename, plus a JSON receipt;
4. rename the claim into `queue/done/`.

A heartbeat thread touches the claim every 20 s. A claim whose heartbeat is older than 10 min
and whose Slurm job is no longer running returns to `pending` (checked by every worker at start
and whenever it finds the queue empty, or by hand with `reap`). Workers therefore tolerate wall
limits, node failures and cancellation: at most one shard of work (seconds to minutes) is lost.

Wall clock: a worker stops claiming 15 min before `SLURM_JOB_END_TIME` and its wrapper submits a
successor. On SIGTERM it returns its claim. A `STOP` file in the run directory ends every worker
after its current shard.

Workers of different GPU types share one queue. `--order asc` takes short proteins first
(A5000), `--order desc` takes long proteins first (H200), so the two meet in the middle. Batch
budgets scale with GPU memory. A CUDA out-of-memory error splits the batch; a single protein
that exceeds a device is recorded in the receipt (`failed`) and leaves a NaN row.

The worker that completes the last shard submits `finalize.sbatch` once (atomic `mkdir` lock).
`finalize` assembles the matrices, audits a random sample against independent float32
single-protein embeddings on CPU, writes `final/MANIFEST.json` and cancels this run's worker
jobs that are still pending.

## Embedding convention

- Model: `Synthyra/ESMplusplus_small` at revision `46c5f7d562e47d4c14165b424c71ab7db008e6fb`
  (17 June 2026; 960-d, 30 layers; the weights blob is identical to the current revision, and the
  code is the generation the Talea v0.3.5 lanes run for ESM++ large). ESM++ small is the Synthyra
  implementation of ESM-C 300M.
- Final hidden state after the last layer norm; arithmetic mean over residue positions only
  (`<cls>`, `<eos>` and padding excluded); float32 accumulation.
- Inference in bfloat16 autocast on GPU (the ESM-C policy of the Spacedust model-scaling runs);
  TF32 is switched off, so float32 reference passes are full precision.
- Whole protein in one pass, no truncation or windowing. `length_of_uid.npy` lets a consumer
  flag proteins beyond the 2,046-residue training length.
- Every worker runs a self-check before it claims work: 32 proteins at length quantiles of the
  data set, float32 one at a time against padded batches of 16 in the run dtype. Gate: minimum
  cosine >= 0.999. The values are in `workers/<worker>.json` and in the manifest.
- Token ids come from a byte lookup table that is asserted equal to the model tokenizer at start.

## Products of a run

| File | Content |
|---|---|
| `final/l2_f32__row.npy` | float32 [rows, 960], L2-normalized, aligned to the input row order (for DPANN the row is the `gid`); NaN = no vector |
| `final/mean_raw_f32__uid.npy` | float32 [unique, 960], raw residue means of the unique sequences |
| `uid_of_row.npy` | int32 [rows], unique-sequence id of each row (-1 = empty sequence) |
| `length_of_uid.npy` | int32 [unique] |
| `final/status__uid.npy` | uint8 [unique], 1 = embedded, 2 = failed |
| `final/MANIFEST.json` | counts, SHA-256, throughput per device, self-checks, audit |
| `out/<shard>.npy`, `out/<shard>.json` | per-shard matrix and receipt (worker, GPU, seconds, residues/s) |

## Commands

```bash
EQ=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue
RUN=$EQ/runs/dpann__esmpp_small__20261111
source $EQ/code/env.sh

# state of a run (inline, cheap)
$PYBIN $EQ/code/embed_queue.py status --run $RUN

# add workers at any time; each is an independent job
sbatch $EQ/code/worker_a5000.sbatch $RUN asc     # -p gpu: one job per user, 12 CPUs
sbatch $EQ/code/worker_h200.sbatch  $RUN desc    # -p gpu_h200 --qos=gpu_h200_4, 28 CPUs
echo <jobid> >> $RUN/worker_jobs.txt             # lets finalize cancel it if it is still pending

touch $RUN/STOP                                  # workers finish their shard and exit
$PYBIN $EQ/code/embed_queue.py reap --run $RUN   # return claims of dead workers by hand
sbatch $EQ/code/finalize.sbatch $RUN             # normally submitted by the last worker
```

A new data set is one `prep` call (see `smoke_cpu.sbatch`) and the same worker scripts.

## Tests

- Model-free queue test (dedup, sharding, four concurrent workers, a claim left by a dead job,
  SIGTERM, finalize against ground truth): passed locally on 11 November 2026.
- `smoke_cpu.sbatch`: 320 sampled DPANN proteins plus 20 duplicate rows through the whole path
  with ESM++ small on CPU in float32, and the same sample through native `transformers` ESM-C
  300M for a cross-implementation comparison. Results are recorded in `docs/redesign/NOTEBOOK.md`.
