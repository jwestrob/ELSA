#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -J wsch_diag

#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_chain/logs/diag_%j.log
# Post hoc diagnostics of accepted anchor pairs (descriptive).
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_chain
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$W/code/production:$W/code/scale:$W/code"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS="${SLURM_CPUS_ON_NODE:?}"
export TMPDIR="$W/tmp" NUMBA_CACHE_DIR="$W/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
MODELS=(esm2_8m esm2_650m esmc600m)
echo "host=$(hostname) cpus=$SLURM_CPUS_ON_NODE job=$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PY" -u "$W/code/diag.py" --threads "$SLURM_CPUS_ON_NODE"
