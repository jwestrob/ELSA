#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J tok_kmeans
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/kmeans_%j.log

# usage: sbatch run_kmeans.sbatch MODEL TRANSFORM KLIST [extra kmeans_tokens.py args]
# Two-level spherical k-means tokens; every K in KLIST is written to $RD/tokens/ when done.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
MODEL=$1; TRANSFORM=$2; KLIST=$3; shift 3
cd "$RD/ws_tokens/code"
echo "$(date -Is) host=$(hostname) threads=$T model=$MODEL transform=$TRANSFORM K=$KLIST extra=$*"
/usr/bin/time -v "$PY" -u kmeans_tokens.py --model "$MODEL" --transform "$TRANSFORM" --K "$KLIST" "$@"
echo "$(date -Is) END"
