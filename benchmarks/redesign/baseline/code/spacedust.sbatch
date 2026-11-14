#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J bl_sd_single
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/baseline_spacedust_%j.log

# Spacedust source207: single-reading scores (engine conserved@2; union of calibrated legacy blocks) next to
# the best-of-many figures quoted in the notebook. evaluate() parallelizes over $SLURM_CPUS_ON_NODE workers.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue TMPDIR="$RD/baseline/tmp"
export NUMBA_CACHE_DIR="$RD/baseline/tmp/numba_${SLURM_JOB_ID}"; mkdir -p "$NUMBA_CACHE_DIR"
/usr/bin/time -v "$PY" -u "$RD/baseline/code/spacedust_single.py" --out "$RD/baseline/results/spacedust" --threads "$T"
