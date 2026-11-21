#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -J search3
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/search3_%j.log
# Locus search, round 3 (search3.py). usage: STAGE=<stage> sbatch --export=ALL [--qos=standard_expand -c 24] run_s3.sbatch
#   lists   control rank lists (gc, wo, genc, zodd, lc) and the gene-level zca10 matrix
#   core    scan-only grid of the main families + cross-validated selection (task 1 AP verdict)
#   ctl     scan-only grid of the controls + selection (task 2)
#   df0 | df1   200-replicate runs, DefenseFinder, half of the run list each
#   x       200-replicate runs, TXSScan and CRISPRCasTyper even genomes (frozen configurations)
#   post    analyse (task 1 recall, generality, 4b) + rates (task 3)
#   p0m3    phase 0 with zca10 (4a) + iterative profile with zca10 centroids (4c)
#   hnsw    HNSW index on zca10 and the k / efSearch sweep (task 5)
#   cli     the one-input command on 24 CPUs (submit with --qos=standard_expand -c 24)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
LS=$RD/locus_search
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue
export TMPDIR=$LS/tmp NUMBA_CACHE_DIR=$LS/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$LS/out3"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID stage=${STAGE:?}"
cd "$LS/code"
case "$STAGE" in
  a)          # task 1 first: grid + selection, then all 200-replicate runs
    "$PY" -u search3.py grid --set main --threads "$T"
    "$PY" -u search3.py select --threads "$T"
    "$PY" -u search3.py final --set df --threads "$T"
    "$PY" -u search3.py final --set x --threads "$T" ;;
  b)          # control lists, control grid, phase 0
    "$PY" -u search3.py lists --threads "$T"
    "$PY" -u search3.py grid --set ctl --threads "$T"
    "$PY" -u search3.py phase0 --threads "$T" ;;
  c)          # needs selection3.json (from a) and the zca10 matrix (from b)
    "$PY" -u search3.py m3 --threads "$T"
    "$PY" -u search3.py hnsw --threads "$T" ;;
  d)          # after a and b
    "$PY" -u search3.py select --threads "$T"
    "$PY" -u search3.py analyse --threads "$T"
    "$PY" -u search3.py rates --threads "$T" ;;
  tau)        # POST HOC: M1z and S1 at tau 0-2 (200-replicate null), then matched-rate recall with tau chosen on odd genomes
    "$PY" -u search3.py final --set tau --threads "$T"
    "$PY" -u search3.py posthoc2 --threads "$T" ;;
  ph)         # POST HOC matched-rate analyses (submit with --qos=standard_expand -c 24: 24 tasks)
    "$PY" -u search3.py posthoc --threads "$T" ;;
  lists)
    "$PY" -u search3.py lists --threads "$T" ;;
  core)
    "$PY" -u search3.py grid --set main --threads "$T"
    "$PY" -u search3.py select --threads "$T" ;;
  ctl)
    "$PY" -u search3.py grid --set ctl --threads "$T"
    "$PY" -u search3.py select --threads "$T" ;;
  df0)
    "$PY" -u search3.py final --set df --part 0/2 --threads "$T" ;;
  df1)
    "$PY" -u search3.py final --set df --part 1/2 --threads "$T" ;;
  x)
    "$PY" -u search3.py final --set x --threads "$T" ;;
  post)
    "$PY" -u search3.py analyse --threads "$T"
    "$PY" -u search3.py rates --threads "$T" ;;
  p0m3)
    "$PY" -u search3.py phase0 --threads "$T"
    "$PY" -u search3.py m3 --threads "$T" ;;
  hnsw)
    "$PY" -u search3.py hnsw --threads "$T" ;;
  cli)
    "$PY" -u interactive.py bench3 --threads "$T" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "$(date -Is) finished"
