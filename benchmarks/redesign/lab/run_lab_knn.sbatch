#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -J lab_knn
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/lab_knn_%j.log
# kNN of the unique sequences of an embed_queue run.
# usage: sbatch --export=ALL[,TRANSFORM=zca10][,K=50][,INDEX=hnsw][,EXACTN=5000][,CHECKNPZ=<earlier kNN npz>] run_lab_knn.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
EMB=${EMB:-/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111}
TAG=${TAG:-esmpp_small}
P=$RD/ws_dpann/package
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID"
CHECK=(); if [[ -n "${CHECKNPZ:-}" ]]; then CHECK=(--check-npz "$CHECKNPZ"); fi
"$PY" -u $RD/lab/code/embed_tokens.py knn --emb "$EMB" --tag "$TAG" --threads "$T" --transform "${TRANSFORM:-zca10}" --k "${K:-50}" \
    --index "${INDEX:-hnsw}" --exact-n "${EXACTN:-0}" "${CHECK[@]}" --work $RD/lab/knn
