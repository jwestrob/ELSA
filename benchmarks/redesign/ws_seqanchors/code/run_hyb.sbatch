#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sa_hyb
#SBATCH --array=0-2%2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors/logs/hyb_%A_%a.log

set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$W/code/production:$W/code/scale_code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_ON_NODE:?}" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$W/tmp" NUMBA_CACHE_DIR="$W/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "host=$(hostname) cpus=$SLURM_CPUS_ON_NODE"; lscpu | grep -E "Model name"
MODELS=(esm2_8m esm2_650m esmc600m)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
/home/jwestrob/miniconda3/envs/elsa/bin/python -u "$W/code/sa.py" arms --workers "$SLURM_CPUS_ON_NODE" \
    --arms "H-ctx60@$M,H-seed60@$M"
