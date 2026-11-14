#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sa_zsearch
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14,node-64-768g-29
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors/logs/zsearch_%j.log

# Exact top-60 inner-product search (faiss knn) of all 67,148 queries vs the whole reference,
# in whitened, learned and raw spaces. node-64-768g-29 is excluded for its stalled reads (ws_metric).
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T"
echo "host=$(hostname) cpus=$T"; lscpu | grep -E "Model name"
/home/jwestrob/miniconda3/envs/elsa/bin/python -u "$W/code/zsearch.py" --threads "$T" \
    --names zca10__esm2_650m,zca10__esmc600m,zlin256__esmc600m,raw__esm2_650m,raw__esmc600m
