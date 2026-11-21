#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J gcenter
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/genome_center/logs/gc_%j.log
# genome_center (gc.py). usage: STAGE=<stage> [ARMS=Z,G1] [PART=i/n] sbatch --export=ALL run_gc.sbatch
#   smoke   Z on 512 member genes + search3 machinery against REPORT3 rows
#   lists   transforms, variance shares, lists and readouts for the arms in $ARMS
#   grid    scan-only grid, 5-fold CV, reproduction gate, primary readout, frozen-configuration scans
#   nullx   200-replicate null runs (CRISPRCasTyper even), part $PART
#   final   result tables and $RD/genome_center/STATUS (FAILED line if the stage crashes)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
GCD=$RD/genome_center
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue
export TMPDIR=$GCD/tmp NUMBA_CACHE_DIR=$GCD/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$GCD/logs" "$GCD/out"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID stage=${STAGE:?}"
cd "$GCD/code"
case "$STAGE" in
  smoke) "$PY" -u gc.py smoke --n 512 --threads "$T" ;;
  lists) "$PY" -u gc.py lists --arms "${ARMS:?}" --threads "$T" ;;
  grid)  "$PY" -u gc.py grid --threads "$T" ;;
  nullx) "$PY" -u gc.py nullx --part "${PART:?}" --threads "$T" ;;
  final)
    if ! "$PY" -u gc.py final --nparts "${NPARTS:-3}" --threads "$T"; then
      echo "FAILED final stage crashed (job $SLURM_JOB_ID; log $GCD/logs/gc_$SLURM_JOB_ID.log)" > "$GCD/STATUS"
      exit 1
    fi ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "$(date -Is) finished"
