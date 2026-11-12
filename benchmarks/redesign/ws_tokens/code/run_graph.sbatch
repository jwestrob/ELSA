#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J tok_graph
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/graph_%j.log

# usage: sbatch run_graph.sbatch MODEL TRANSFORM RESOLUTIONS [extra graph_tokens.py args]
# Mutual-kNN graph + Leiden communities (workstream venv with igraph/leidenalg).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
MODEL=$1; TRANSFORM=$2; RES=$3; shift 3
cd "$RD/ws_tokens/code"
echo "$(date -Is) host=$(hostname) threads=$T model=$MODEL transform=$TRANSFORM res=$RES extra=$*"
/usr/bin/time -v "$PY" -u graph_tokens.py --model "$MODEL" --transform "$TRANSFORM" --resolution "$RES" "$@"
echo "$(date -Is) END"
