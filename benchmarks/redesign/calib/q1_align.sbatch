#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J calib_q1
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/calib_q1_%j.log
# ws_calib Q1: sample pairs, align them explicitly (MMseqs2 align, Smith-Waterman, E <= 10 reported), classify.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
W=$RD/calib/work/q1; U=$RD/subgenic/db/uniqdb
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_calib_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
if [[ -z "${SKIP_PAIRS:-}" ]]; then
  "$PY" -u "$RD/calib/code/q1.py" pairs --threads "$T"
  "$MM" rmdb "$W/pref" -v 1 2>/dev/null || true; "$MM" rmdb "$W/aln" -v 1 2>/dev/null || true
  rm -f "$W/aln.parquet" "$W/aln.tsv"
  "$MM" tsv2db "$W/pref.tsv" "$W/pref" --output-dbtype 7 -v 2
  stamp "START align"
  "$MM" align "$U" "$U" "$W/pref" "$W/aln" -e 10 --alignment-mode 3 --threads "$T" -v 2
  stamp "DONE align"
  "$MM" convertalis "$U" "$U" "$W/aln" "$W/aln.tsv" \
     --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
fi
"$PY" -u "$RD/calib/code/q1.py" classify --threads "$T"
stamp "done"
