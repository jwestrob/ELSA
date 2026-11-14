#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -J lab_hybrid
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/lab_hybrid_%j.log
# Hybrid tokens: families merged by centroid similarity. usage: sbatch --export=ALL,BASE=<family parquet>,BASETAG=<tag>,CT=<centroid transform>[,RESQ=..][,RES=..] run_lab_hybrid.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
EMB=${EMB:-/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111}
TAG=${TAG:-esmpp_small}
P=$RD/ws_dpann/package
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID"
"$PY" -u $RD/lab/code/embed_tokens.py hybrid --emb "$EMB" --tag "$TAG" --threads "$T" --base "${BASE:?}" --base-tag "${BASETAG:?}" --universe $P/universe/genes.parquet --centroid-transform "${CT:-genomecentred}" --res-q "${RESQ-0.5,0.9,0.99,0.999}" --res "${RES-}" --out $RD/engine/dpann_tokens
