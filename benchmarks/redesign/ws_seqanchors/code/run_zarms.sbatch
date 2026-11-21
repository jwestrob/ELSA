#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sa_zarms
#SBATCH --array=0-2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors/logs/zarms_%A_%a.log

set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$W/code/production:$W/code/scale_code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_ON_NODE:?}" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$W/tmp" NUMBA_CACHE_DIR="$W/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "host=$(hostname) cpus=$SLURM_CPUS_ON_NODE"; lscpu | grep -E "Model name"
ARMS=("Z-seed60@esm2_650m,HZ-seed60@esm2_650m,R-seed60@esm2_650m"
      "Z-seed60@esmc600m,HZ-seed60@esmc600m,R-seed60@esmc600m"
      "L-seed60@esmc600m,HL-seed60@esmc600m")
/home/jwestrob/miniconda3/envs/elsa/bin/python -u "$W/code/sa.py" arms --workers "$SLURM_CPUS_ON_NODE" \
    --arms "${ARMS[$SLURM_ARRAY_TASK_ID]}"
