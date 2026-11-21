#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J vocab
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/vocab/logs/vocab_big_%j.log
# Runs one vocab-workstream script with all allocated CPUs (process pool or numba threads).
# usage: SCRIPT=vocab.py ARGS="ladder --arm main+genus+genome" sbatch --export=ALL run_big.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_vocab_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) host $(hostname) cpus $T script ${SCRIPT:?} args ${ARGS:-}"
cd "$RD/vocab/code"
"$PY" -u "$RD/vocab/code/${SCRIPT}" ${ARGS:-} --threads "$T"
echo "$(date -Is) done"
