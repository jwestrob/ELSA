#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ctxk_edges
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/ctxk_edges_%j.log

# Context kernel, stage 1: candidate edges, context scores, null, FDR support, gold diagnostics.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" MKL_NUM_THREADS="$T"
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_ctxk_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
"$PY" -u "$RD/context_kernel/code/context_kernel.py" edges --cap "${CAP:-64}" --alpha 0.01 --seed 17
