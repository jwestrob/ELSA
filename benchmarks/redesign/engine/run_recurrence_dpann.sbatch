#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J rd_recur_dpann
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/recur_dpann_%j.log

# Recurrence engine on the DPANN package with sequence-family tokens, then the gold-locus scorer.
# usage: sbatch --export=ALL,W=<window> run_recurrence_dpann.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
W="${W:-3}"
ARM=dpann__mmseqs__seq__id30c80
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
"$PY" -u "$RD/engine/code/recurrence.py" --arm "$ARM" --universe "$P/universe" \
    --tokens "$P/tokens/mmseqs__seq__id30c80__dpann.parquet" --w "$W" --null "${NULL:-genome}" --benchmark none --all-genome-loci \
    --out "$RD/engine/${OUTDIR:-results_dpann}" --threads "$T"
O=$RD/engine/${OUTDIR:-results_dpann}/${ARM}__w${W}
"$PY" -u "$P/scorer/score_dpann_loci.py" --pkg "$P" --loci "$O/loci_all_genomes.parquet" \
    --out "$O/score_min2" --threads "$T" --min-members 2
"$PY" -u "$P/scorer/score_dpann_loci.py" --pkg "$P" --loci "$O/loci_all_genomes.parquet" \
    --out "$O/score_min2_span30" --threads "$T" --min-members 2 --max-span 30
