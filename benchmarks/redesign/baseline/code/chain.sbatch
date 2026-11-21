#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J bl_chain
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/baseline_chain_%j.log

# Legacy ELSA on DPANN, one arm: [soft-ZCA vectors] -> kNN + chaining -> 4 re-chainings of the saved
# anchors (identity replay, permutation seeds 17, 18, 19).
# usage: ARM=raw|zca [BL_SUBSET=N] sbatch --export=ALL [--qos=.. -c ..] chain.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
C=$RD/baseline/code
T="${SLURM_CPUS_ON_NODE:?}"
ARM="${ARM:?}"
# one OpenMP-capable runtime at full width at a time: BLAS only in the transform step; hnswlib uses
# std::thread (pinned to T in legacy.py); numba uses the workqueue layer with T threads
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/baseline/tmp" NUMBA_CACHE_DIR="$RD/baseline/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "job ${SLURM_JOB_ID} arm $ARM threads $T subset ${BL_SUBSET:-0} host $(hostname) $(date)"
if [[ "$ARM" == zca && ! -f "$RD/baseline/vectors/esmpp_small__soft_zca_a0.1.npy" ]]; then
    OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" /usr/bin/time -v "$PY" -u "$C/legacy.py" zca
fi
/usr/bin/time -v "$PY" -u "$C/legacy.py" chain --arm "$ARM"
for s in identity 17 18 19; do
    /usr/bin/time -v "$PY" -u "$C/legacy.py" perm --arm "$ARM" --seed "$s"
done
echo "done $(date)"
