#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J bl_diag
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/baseline_diag_%j.log

# Origin of the permuted-order blocks (DuckDB with $SLURM_CPUS_ON_NODE threads).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export TMPDIR="$RD/baseline/tmp" PYTHONDONTWRITEBYTECODE=1
/usr/bin/time -v "$PY" -u "$RD/baseline/code/diag_null.py" --threads "$T" --out "$RD/baseline/results/diag_null.json"
