#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J rd_ctx_big
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/context_units/logs/ctxbig_%j.log
# Test 2 (embedding-only kernel with contexts). usage: sbatch run_big.sbatch <bash step file>  (uses $T, $PY, $C)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
export RD
export T="${SLURM_CPUS_ON_NODE:?}"
export PY=$RD/ws_tokens/env/bin/python
export C=$RD/context_units/code
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_ctxbig_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "job $SLURM_JOB_ID host $(hostname) cpus $T step $1 start $(date -u +%FT%TZ)"
bash -euo pipefail "$1"
echo "end $(date -u +%FT%TZ)"
