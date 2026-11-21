#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J bl_clust
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/baseline_cluster_%j.log

# Legacy ELSA on DPANN, one arm: run_chain_pipeline resumed from the blocks checkpoint (overlap
# clustering with its threaded interval sweep, contained-cluster merge, CSV outputs).
# usage: ARM=raw|zca [BL_SUBSET=N] sbatch --export=ALL cluster.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
C=$RD/baseline/code
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/baseline/tmp" NUMBA_CACHE_DIR="$RD/baseline/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "job ${SLURM_JOB_ID} arm ${ARM:?} threads $T subset ${BL_SUBSET:-0} host $(hostname) $(date)"
/usr/bin/time -v "$PY" -u "$C/legacy.py" cluster --arm "$ARM"
echo "done $(date)"
