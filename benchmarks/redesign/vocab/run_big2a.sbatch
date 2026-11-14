#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J vocab2a
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/vocab/logs/vocab_big2a_%j.log
# Round 2A: runs a ';'-separated command sequence with all 64 allocated CPUs.
# usage: CMDS="vocab2a.py synS;vocab2a.py ladder ..." sbatch --export=ALL run_big2a.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_vocab_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
cd "$RD/vocab/code"
echo "$(date -Is) host $(hostname) cpus $T cmds ${CMDS:?}"
IFS=';' read -ra parts <<< "$CMDS"
for c in "${parts[@]}"; do
  echo "$(date -Is) >>> $c"
  "$PY" -u $c --threads "$T"
done
echo "$(date -Is) done"
