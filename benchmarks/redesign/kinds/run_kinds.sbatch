#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J kinds
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/kinds_%j.log
# usage: export STEPS="prep check main bench"; sbatch --export=ALL run_kinds.sbatch   (space-separated steps)
# every step is numba-parallel over $SLURM_CPUS_ON_NODE threads
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/kinds/tmp" NUMBA_CACHE_DIR="$RD/kinds/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
cd "$RD/kinds/code"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID steps=${STEPS:?}"
for st in $STEPS; do
    echo "$(date -Is) STEP $st"
    case $st in
        check) "$PY" -u kinds_check.py "$T" ;;
        prep|main|bench) "$PY" -u kinds.py "$st" --threads "$T" ;;
        bench_r*) "$PY" -u kinds.py bench --unit "${st#bench_}" --threads "$T" ;;
        withheld) "$PY" -u kinds.py withheld --threads "$T" --levels "${LEVELS:-full,L1,L2,L2ctl}" ;;
        *) "$PY" -u $st --threads "$T" ;;
    esac
done
echo "$(date -Is) done"
