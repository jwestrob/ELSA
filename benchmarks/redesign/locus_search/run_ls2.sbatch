#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -J locus_search2
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/locus_search2_%j.log
# Locus search, round 2. usage: STAGE=<stage> sbatch --export=ALL [--qos=standard_expand -c 24] run_ls2.sbatch
#   core           evidence prep, CV grid (S2/S3, all DefenseFinder queries), final 200-replicate runs (dfall)
#   xeven          final 200-replicate S2/S3 runs, TXSScan and CRISPRCasTyper even genomes
#   rategrid_odd | rategrid_even   rate-constrained candidate grid (40-replicate fitted-tail null)
#   analyse        unlabelled-hit evidence (task 1) and fast-null calibration (task 4)
#   rate           rate-constrained selection (after both rate grids)
#   pairlib        odd-genome token-pair library on even genomes (task 6)
#   interactive    HNSW / indexed MMseqs2 path (task 5)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
LS=$RD/locus_search
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue
export TMPDIR=$LS/tmp NUMBA_CACHE_DIR=$LS/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$LS/out2"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID stage=${STAGE:?}"
cd "$LS/code"
case "$STAGE" in
  core)
    "$PY" -u round2.py cvgrid --threads "$T"
    "$PY" -u round2.py final --set dfall --threads "$T" ;;
  evprep)
    "$PY" -u evidence.py prep --threads "$T" ;;
  xeven)
    "$PY" -u round2.py final --set xeven --threads "$T" ;;
  rategrid_odd)
    "$PY" -u round2.py rategrid --split odd --threads "$T" ;;
  rategrid_even)
    "$PY" -u round2.py rategrid --split even --threads "$T" ;;
  analyse)
    "$PY" -u evidence.py analyse --threads "$T"
    "$PY" -u fastnull.py calib --sets dfall --threads "$T" ;;
  calib2)
    "$PY" -u fastnull.py calib --sets dfall,xeven --threads "$T" ;;
  shufctl)
    "$PY" -u round2.py shufctl --threads "$T" ;;
  rate)
    "$PY" -u evidence.py analyse --threads "$T"
    "$PY" -u evidence.py rate --threads "$T" ;;
  pairlib)
    "$PY" -u evidence.py prep --threads "$T" || echo "evidence prep failed"
    "$PY" -u pairlib.py --threads "$T" ;;
  interactive)
    "$PY" -u interactive.py ${INTERACTIVE_ARGS:-all} --threads "$T" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "$(date -Is) finished"
