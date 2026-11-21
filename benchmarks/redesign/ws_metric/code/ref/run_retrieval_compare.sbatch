#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J elsa_retrcmp
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/seq_invisible_20261111/logs/retrcmp_%j.log

set -euo pipefail
ROOT=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust
SCALE="$ROOT/model_scaling_20260913"
RUN="$ROOT/seq_invisible_20261111"
N="$ROOT/native_run_20260910"
T="${SLURM_CPUS_ON_NODE:?}"
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
K="$RUN/work_retrieval"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCALE/production:$SCALE/code"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RUN/tmp" NUMBA_CACHE_DIR="$RUN/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"

"$PY" -u "$RUN/code/retrieval_compare.py" prep --run "$RUN" --threads "$T"
rm -rf "$K/db"; mkdir -p "$K/db"
"$MM" tsv2db "$K/pref.tsv" "$K/db/pref" --output-dbtype 7 -v 2
"$MM" align "$N/query/querydb" "$N/reference/keggclusterdb" "$K/db/pref" "$K/db/aln_seq" \
    -e inf --alignment-mode 3 --threads "$T" -v 2
"$MM" prefixid "$K/db/aln_seq" "$K/aln_seq.tsv" --tsv --threads "$T" -v 2
"$PY" -u "$RUN/code/retrieval_compare.py" summarize --run "$RUN" --threads "$T"
