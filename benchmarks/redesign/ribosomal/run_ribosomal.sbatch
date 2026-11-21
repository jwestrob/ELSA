#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J rd_ribo
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/ribosomal_%j.log

# Ribosomal-locus check.
#   MODE=gold   build the gold runs and controls, then check the base arm (per-run table kept)
#   MODE=all    check every arm directory listed in $RD/ribosomal/arms.txt, P arms at a time
#   MODE=diag   diagnosis of the base arm (diagnose_base.py)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
C=$RD/ribosomal/code
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue NUMBA_CACHE_DIR="$RD/ribosomal/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR" "$RD/ribosomal/arms"
BASE=$RD/engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3
tag () { echo "$(basename "$(dirname "$1")")__$(basename "$1")"; }
case "${MODE:?}" in
  gold)
    "$PY" -u "$C/ribosomal_check.py" gold --threads "$T"
    /usr/bin/time -v "$PY" -u "$C/ribosomal_check.py" check "$BASE" --out "$RD/ribosomal/arms/$(tag "$BASE")" \
        --threads "$T" --per-run
    ;;
  all)
    # P concurrent arms with T/P numba threads each: P * (T/P) = T
    P=6; TH=$(( T / P ))
    export C PY RD TH
    grep -v '^#' "$RD/ribosomal/arms.txt" | xargs -P "$P" -I{} bash -c \
      'd={}; t="$(basename "$(dirname "$d")")__$(basename "$d")"; "$PY" -u "$C/ribosomal_check.py" check "$d" --out "$RD/ribosomal/arms/$t" --threads "$TH" --per-run > "$RD/ribosomal/arms/$t.line" 2> "$RD/ribosomal/arms/$t.err" || echo "FAILED $d" >> "$RD/ribosomal/arms/FAILED.txt"'
    cat "$RD"/ribosomal/arms/*.line
    [[ -s "$RD/ribosomal/arms/FAILED.txt" ]] && cat "$RD/ribosomal/arms/FAILED.txt" || true
    "$PY" -u "$C/aggregate.py"
    ;;
  diag)
    "$PY" -u "$C/diagnose_base.py" --threads "$T"
    ;;
esac
