#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J wsm_align
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric/logs/align_%j.log

# usage: sbatch --export=ALL,TAG=full[,SKIP_ALIGN=1] run_align.sbatch
# Pairwise Smith-Waterman on the union of retrieved pairs (recipe of $S/code/pipeline.sh),
# then all metrics.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric
N=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/native_run_20260910
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
TAG="${TAG:-full}"
A="$W/$TAG/align"
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" PYTHONDONTWRITEBYTECODE=1
cd "$W/code"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T tag=$TAG"
# popB pair scores for any evaluated model that lacks them (popB.parquet arrived after its eval)
if [ "$TAG" = "full" ] && [ -f "$W/sets/popB.parquet" ]; then
  for m in esm2_8m esm2_650m esmc600m esmc6b; do
    f="$W/$TAG/eval/$m/pairs_popB.parquet"
    has_raw=$("$PY" -c "import sys,pyarrow.parquet as pq; print(int('raw' in pq.read_schema(sys.argv[1]).names))" "$f" 2>/dev/null || echo 0)
    if [ -f "$W/$TAG/eval/$m/eval_info.json" ] && [ "$has_raw" != "1" ]; then
      stamp "pairs popB $m"
      "$PY" -u run_model.py --model "$m" --stages pairs --pops popB --threads "$T"
    fi
  done
fi
if [ -z "${SKIP_ALIGN:-}" ]; then
  "$PY" -u summarize.py collect --threads "$T" --tag "$TAG"
  rm -rf "$A/db"; mkdir -p "$A/db"
  stamp "START mmseqs_align"
  "$MM" tsv2db "$A/pref.tsv" "$A/db/pref" --output-dbtype 7 -v 2
  "$MM" align "$N/query/querydb" "$N/reference/keggclusterdb" "$A/db/pref" "$A/db/aln" \
      -e inf --alignment-mode 3 --threads "$T" -v 2
  "$MM" prefixid "$A/db/aln" "$A/aln.tsv.tmp" --tsv --threads "$T" -v 2
  mv "$A/aln.tsv.tmp" "$A/aln.tsv"
  stamp "DONE mmseqs_align"
  wc -l "$A/pref.tsv" "$A/aln.tsv"
fi
"$PY" -u summarize.py summarize --threads "$T" --tag "$TAG"
stamp "DONE summarize"
"$PY" -u report_tables.py --tag "$TAG"
