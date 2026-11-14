#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J wsm_popb8m
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14,node-64-768g-29
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric/logs/popb8m_%j.log

# One slot, sequential: popB set -> popB pair scores for esm2_8m -> align + summarize what exists.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric
T="${SLURM_CPUS_ON_NODE:?}"
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" PYTHONDONTWRITEBYTECODE=1
cd "$W/code"
echo "host=$(hostname) threads=$T"
"$PY" -u prep_sets.py --threads "$T" --only popB
"$PY" -u run_model.py --model esm2_8m --stages pairs --pops popB --threads "$T"
bash run_align.sbatch
