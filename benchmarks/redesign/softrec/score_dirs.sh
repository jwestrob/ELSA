#!/bin/bash
# Score engine-format result directories: DPANN gold scorer (min 2 members), paired comparisons against the
# MMseqs2 base and the rescue3 default (genome bootstrap; odd = tune, even = test), ribosomal check.
# usage: score_dirs.sh <threads> <result dir>...   (only tune-split summed excess is echoed to the log)
set -uo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
BASE=$RD/engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3
DEF=$RD/engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3
T=$1; shift
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$RD/tmp/numba_softrec_score_$$}"
mkdir -p "$NUMBA_CACHE_DIR"
for O in "$@"; do
    [[ -f "$O/loci_all_genomes.parquet" ]] || { echo "SKIP $O (no loci)"; continue; }
    t0=$(date +%s)
    if [[ ! -f "$O/score_min2/instance_scores.parquet" ]]; then
        "$PY" -u "$P/scorer/score_dpann_loci.py" --pkg "$P" --loci "$O/loci_all_genomes.parquet" \
            --out "$O/score_min2" --threads "$T" --min-members 2 > "$O/score_min2.log" 2>&1 || echo "SCORER failed $O"
    fi
    "$PY" -u "$RD/engine/code/dpann_compare.py" "$O" "$BASE" > "$O/compare_vs_base.txt" 2>&1
    "$PY" -u "$RD/engine/code/dpann_compare.py" "$O" "$DEF" > "$O/compare_vs_default.txt" 2>&1
    echo "SCORED $(basename "$O") $(grep 'tune (odd)  DefenseFinder+TXSScan' "$O/compare_vs_base.txt" | sed 's/^ *//') [$(( $(date +%s) - t0 ))s]"
    "$PY" -u "$RD/ribosomal/code/ribosomal_check.py" check "$O" --threads "$T" > "$O/ribosomal_check.log" 2>&1 || echo "RIBOSOMAL check failed $O"
    grep RIBOSOMAL "$O/ribosomal_check.log" | sed -E 's/\| resolved.*//'
done
