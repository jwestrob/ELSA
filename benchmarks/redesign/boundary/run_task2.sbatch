#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J bnd_t2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/boundary_t2_%j.log

# Task 2: merged universe (split pairs as one position) -> engine (rescue3 tokens, w3) -> scorer -> paired comparison.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_bnd_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
M=$RD/boundary/merged
ARM=dpann__mmseqs__seq__e3c50s75_rescue3_merged
echo "host $(hostname) cpus $T"
"$PY" -u "$RD/boundary/code/merge_universe.py" --threads "$T"
"$PY" -u "$RD/engine/code/recurrence.py" --arm "$ARM" --universe "$M/pkg/universe" --tokens "$M/tokens_rescue3_merged.parquet" \
    --w 3 --null genome --benchmark none --all-genome-loci --out "$M/results" --threads "$T"
O=$M/results/${ARM}__w3
"$PY" -u "$P/scorer/score_dpann_loci.py" --pkg "$M/pkg" --loci "$O/loci_all_genomes.parquet" --out "$O/score_min2" \
    --threads "$T" --min-members 2 > "$O/score_min2.stdout"
"$PY" -u "$RD/boundary/code/task2_compare.py" --threads "$T" --merged-result "$O" --rule 'D:{"core": "F", "level": 200}'
