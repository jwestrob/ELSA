#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -J lab_graph
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/lab_graph_%j.log
# Graph tokens: Leiden communities of the mutual-kNN graph. usage: sbatch --export=ALL,KNN=<npz>,GTAG=<tag>[,K=20][,MOD=..][,CPMQ=..][,MINQ=..] run_lab_graph.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
EMB=${EMB:-/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111}
TAG=${TAG:-esmpp_small}
P=$RD/ws_dpann/package
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID"
"$PY" -u $RD/lab/code/embed_tokens.py graph --emb "$EMB" --tag "${GTAG:?}" --threads "$T" --knn "${KNN:?}" --k "${K:-20}" --modularity "${MOD-100,300,1000,3000}" --cpm-q "${CPMQ-}" --min-cos-q "${MINQ:-0}" --out $RD/engine/dpann_tokens
