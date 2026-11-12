#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sr_knn
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/softrec_knn_%j.log
# Deep HNSW kNN of the unique sequences (softrec). usage: sbatch --export=ALL,SOURCE=zca10|head[,K=2000] run_knn.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID source=${SOURCE:?} k=${K:-2000}"
"$PY" -u "$RD/softrec/code/knn_deep.py" --source "$SOURCE" --k "${K:-2000}" --threads "$T"
echo "$(date -Is) done"
