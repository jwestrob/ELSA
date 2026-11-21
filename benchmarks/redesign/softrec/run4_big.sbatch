#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sr4_big
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/softrec4_big_%j.log
# softrec4 steps that hold the deep lists (statistics, candidates, support edges) or large graphs (Leiden).
# usage: export STEPS="softrec4.py stats --modes 0,1,2 --then-cands ...;readings4.py graph ..."; sbatch --export=ALL run4_big.sbatch
#        (each step = a script under $RD/softrec/r4/code plus its arguments; --threads is appended)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/softrec/r4/tmp" NUMBA_CACHE_DIR="$RD/softrec/r4/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID"
IFS=';' read -r -a S <<< "${STEPS:?}"
for st in "${S[@]}"; do
    [[ -z "${st// }" ]] && continue
    echo "$(date -Is) STEP $st"
    if [[ "$st" == readings4.py* ]]; then B=1; else B=$T; fi     # the scorer's process pool runs with BLAS threads 1
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=$B OPENBLAS_NUM_THREADS=$B MKL_NUM_THREADS=$B "$PY" -u "$RD/softrec/r4/code/"$st --threads "$T"
done
echo "$(date -Is) done"
