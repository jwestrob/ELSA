#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J wsm_model
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric/logs/model_%j.log

# usage: sbatch --export=ALL,MODEL=esm2_8m,STAGES=stats+train+eval[,SMOKE=1][,ONLY=arm1+arm2] run_model.sbatch
# (sbatch --export splits on commas, so lists use "+")
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" PYTHONDONTWRITEBYTECODE=1
cd "$W/code"
echo "host=$(hostname) threads=$T model=$MODEL stages=$STAGES smoke=${SMOKE:-0} only=${ONLY:-}"
/home/jwestrob/miniconda3/envs/elsa/bin/python -u run_model.py --model "$MODEL" --stages "${STAGES//+/,}" \
    --threads "$T" ${SMOKE:+--smoke} ${ONLY:+--only "${ONLY//+/,}"} ${POPS:+--pops "${POPS//+/,}"}
