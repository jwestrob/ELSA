#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J tok_py
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/py_%j.log

# usage: sbatch run_py.sbatch SCRIPT.py [args]   (the script receives --threads $SLURM_CPUS_ON_NODE)
set -euo pipefail
WS=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens
PY=${PYBIN:-/home/jwestrob/miniconda3/envs/elsa/bin/python}
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
cd "$WS/code"
S=$1; shift
echo "$(date -Is) host=$(hostname) threads=$T script=$S args=$*"
/usr/bin/time -v "$PY" -u "$S" --threads "$T" "$@"
echo "$(date -Is) END"
