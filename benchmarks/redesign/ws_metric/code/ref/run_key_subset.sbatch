#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J elsa_keysubset
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/seq_invisible_20261111/logs/keysubset_%j.log

# TM-align on the decisive subset: sequence-invisible pairs in calibrated blocks, plus controls.
set -euo pipefail
ROOT=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust
RUN="$ROOT/seq_invisible_20261111"
N="$ROOT/native_run_20260910"
T="${SLURM_CPUS_ON_NODE:?}"
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
QFS="$N/query/querydb_foldseek"
TFS="$N/reference/keggclusterdb_foldseek"
K="$RUN/work_key"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }

stamp "START prep"
"$PY" -u "$RUN/code/key_subset.py" prep --run "$RUN" --threads "$T"
stamp "DONE prep"
rm -rf "$K/db"; mkdir -p "$K/db"
"$MM" tsv2db "$K/pref_key.tsv" "$K/db/pref_key" --output-dbtype 7 -v 2
stamp "START foldseek_tmalign"
"$FS" tmalign "$QFS" "$TFS" "$K/db/pref_key" "$K/db/aln_tm" --tmscore-threshold 0.0 -a 1 --threads "$T" -v 2
stamp "DONE foldseek_tmalign"
# The aliased target DB (headers carry K<key>) was built by the main job.
"$FS" convertalis "$QFS" "$RUN/work/db/tref/tref" "$K/db/aln_tm" "$K/aln_tm.tsv" \
    --format-output query,target,qtmscore,ttmscore,alntmscore,lddt,fident,qcov,tcov,alnlen,qlen,tlen,evalue,bits \
    --exact-tmscore 1 --threads "$T" -v 2
stamp "DONE tm_convertalis"
"$PY" -u "$RUN/code/key_subset.py" summarize --run "$RUN" --threads "$T"
stamp "DONE summarize"
