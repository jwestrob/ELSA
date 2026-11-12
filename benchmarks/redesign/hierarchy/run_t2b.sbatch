#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J hier_t2b
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/hier_t2b_%j.log
# Task 2, second half: ribosomal guard on the candidate runs, guarded / unguarded selection on odd genomes,
# the 'both' sets (B1, B2) and nested versions of the chosen embedding set, then the final paired comparisons.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
C=$RD/hierarchy/code
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue TMPDIR="$RD/tmp"
echo "$(date -Is) host=$(hostname) cpus=$T job=$SLURM_JOB_ID"
export NUMBA_NUM_THREADS=$T NUMBA_CACHE_DIR=$RD/tmp/numba_${SLURM_JOB_ID}_ribo; mkdir -p "$NUMBA_CACHE_DIR"
for s in E1 E2 E3 E4 E5 E6 S1 S2a S2b S2c; do
    d=$RD/engine/results_dpann_hier/hier__ml__${s}__w3
    [[ -f $d/ribosomal_check.json ]] || "$PY" -u $RD/ribosomal/code/ribosomal_check.py check $d --threads $T 2>/dev/null < /dev/null | sed "s/^/$s /"
done
"$PY" -u $C/select_t2.py
"$PY" -u $C/nested.py $(cat $C/nest_sets.txt)
N=$(grep -c . $C/sets_B.tsv); PAR=$(( N < 4 ? N : 4 )); TH=$(( T / PAR ))
grep -v '^#' $C/sets_B.tsv | awk 'NF' | xargs -P "$PAR" -L 1 bash -c 'bash '"$C"'/one_ml2.sh "$0" "$1" '"$TH"
"$PY" -u $C/select_t2.py --final
echo "$(date -Is) end"
