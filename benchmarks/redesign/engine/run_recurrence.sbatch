#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J rd_recur
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/recur_%j.log

# usage: sbatch --export=ALL,ARM=<token arm>,W=<window>[,ORIENTED=1][,OUTDIR=<results dir name>] run_recurrence.sbatch
set -euo pipefail
ROOT=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust
SCALE="$ROOT/model_scaling_20260913"
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCALE/production:$SCALE/code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_ON_NODE:?}" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
EXTRA=()
if [[ "${ORIENTED:-0}" == 1 ]]; then EXTRA+=(--oriented); fi
/home/jwestrob/miniconda3/envs/elsa/bin/python -u "$RD/engine/code/recurrence.py" \
    --arm "${ARM:?}" --w "${W:-3}" --null "${NULL:-genome}" --out "$RD/engine/${OUTDIR:-results}" --threads "$SLURM_CPUS_ON_NODE" "${EXTRA[@]}"
