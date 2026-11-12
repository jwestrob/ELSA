#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J rd_recur_dp
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/recur_dpann_%j.log

# Recurrence engine on the DPANN package with any token arm, then the gold-locus scorer.
# usage: sbatch --export=ALL,ARM=<arm name>,TOKENS=<gid,token parquet>,W=<window>[,OUTDIR=..][,NULL=..] run_recurrence_dpann_arm.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
W="${W:-3}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
OUT=$RD/engine/${OUTDIR:-results_dpann_v2}
EXTRA=(); WTAG=$W
if [[ -n "${ADAPTIVE:-}" ]]; then EXTRA+=(--adaptive "$ADAPTIVE"); WTAG=a${ADAPTIVE##*:}; fi   # window grid, e.g. 1:2:3:5:8
"$PY" -u "$RD/engine/code/recurrence.py" --arm "${ARM:?}" --universe "$P/universe" --tokens "${TOKENS:?}" --w "$W" \
    --null "${NULL:-genome}" --benchmark none --all-genome-loci --out "$OUT" --threads "$T" "${EXTRA[@]}"
O=$OUT/${ARM}__w${WTAG}
"$PY" -u "$P/scorer/score_dpann_loci.py" --pkg "$P" --loci "$O/loci_all_genomes.parquet" \
    --out "$O/score_min2" --threads "$T" --min-members 2
"$PY" -u "$P/scorer/score_dpann_loci.py" --pkg "$P" --loci "$O/loci_all_genomes.parquet" \
    --out "$O/score_min2_span30" --threads "$T" --min-members 2 --max-span 30
