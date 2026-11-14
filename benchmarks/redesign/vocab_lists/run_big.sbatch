#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J vl_big
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/vocab_lists/logs/vl_big_%j.log
# ws_vocab2b: learner jobs (lists, round-0 ladders, full runs) and scoring steps with all allocated CPUs.
# usage: ARGS="job --excl genome --arms syn,ctx" sbatch --export=ALL run_big.sbatch            (one vl.py command)
#    or: CMDS="vl_score.py refcheck;vl.py job --excl genome --arms syn" sbatch --export=ALL run_big.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/vocab_lists/tmp" NUMBA_CACHE_DIR="$RD/vocab_lists/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
cd "$RD/vocab_lists/code"
if [[ -n "${ARGS:-}" ]]; then CMDS="vl.py ${ARGS}"; fi
echo "$(date -Is) host $(hostname) cpus $T cmds ${CMDS:?}"
IFS=';' read -ra parts <<< "$CMDS"
for c in "${parts[@]}"; do
  [[ -z "${c// }" ]] && continue
  echo "$(date -Is) >>> $c"
  if [[ "$c" == vl_score.py* && "$c" != *typing2* ]]; then B=1; else B=$T; fi   # scorer pool: BLAS 1; faiss kNN: all
  # shellcheck disable=SC2086
  OMP_NUM_THREADS=$B OPENBLAS_NUM_THREADS=$B MKL_NUM_THREADS=$B "$PY" -u $c --threads "$T"
done
echo "$(date -Is) done"
