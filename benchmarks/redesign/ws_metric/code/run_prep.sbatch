#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J wsm_prep
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric/logs/prep_%j.log

set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" PYTHONDONTWRITEBYTECODE=1
cd "$W/code"
echo "host=$(hostname) threads=$T only=${ONLY:-all}"
/home/jwestrob/miniconda3/envs/elsa/bin/python -u prep_sets.py --threads "$T" ${ONLY:+--only "$ONLY"}
