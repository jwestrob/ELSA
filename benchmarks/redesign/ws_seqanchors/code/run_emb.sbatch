#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sa_emb
#SBATCH --array=0-2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors/logs/emb_%A_%a.log

# One task per model: context60 replay + regeneration under three query-order permutations,
# then E-ctx60/{cos,rank} and E-seed60/{cos,rank}, each with observed + three permuted chains.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors
MODELS=(esm2_8m esm2_650m esmc600m)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$W/code/production:$W/code/scale_code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_ON_NODE:?}" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$W/tmp" NUMBA_CACHE_DIR="$W/tmp/numba_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "host=$(hostname) model=$M cpus=$SLURM_CPUS_ON_NODE"
lscpu | grep -E "Model name"
/home/jwestrob/miniconda3/envs/elsa/bin/python -u "$W/code/sa.py" emb --model "$M" --workers "$SLURM_CPUS_ON_NODE"
