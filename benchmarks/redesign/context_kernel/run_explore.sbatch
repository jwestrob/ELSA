#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ctxk_expl
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/ctxk_expl_%j.log

# Exploratory controls (not part of the predeclared selection):
#  1. context scores with gene order permuted within genomes (64 numba threads)
#  2. Leiden: V2b = dense_w1-supported edges, sim weight; permtop = top-K edges by permuted-order dense_w1
#     score (K = true support count), sim and sim x ctx weights. Single-threaded Leiden runs side by side.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
C=$RD/context_kernel/code/context_kernel.py
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T"
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_ctxk_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
"$PY" -u "$C" permctx --seed 18
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -u "$C" families --edge-set sup --variant dense_w1 --weights sim --res 10,100 --procs 2 --seed 17 &
p1=$!
"$PY" -u "$C" families --edge-set permtop --variant dense_w1 --weights sim,simctx --res 10,100 --procs 4 --seed 17 &
p2=$!
wait $p1; wait $p2
