#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sr2_big
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/softrec2_big_%j.log
# softrec2 steps that hold the deep lists in memory (statistics, candidates, validated edges) or large graphs (Leiden).
# usage: export STEPS="softrec2.py stats --then-cands --edges --edge-names a,b;..."; sbatch --export=ALL run2_big.sbatch
#        (each step is a script under $RD/softrec/code plus its arguments; --threads is appended)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_softrec2_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID"
IFS=';' read -r -a S <<< "${STEPS:?}"
for st in "${S[@]}"; do
    [[ -z "${st// }" ]] && continue
    echo "$(date -Is) STEP $st"
    # shellcheck disable=SC2086
    "$PY" -u "$RD/softrec/code/"$st --threads "$T"
done
echo "$(date -Is) done"
