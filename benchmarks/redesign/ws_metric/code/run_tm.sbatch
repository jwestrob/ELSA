#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J wsm_tm
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric/logs/tm_%j.log

# TM-align of a hash sample of sequence-invisible retrieved pairs ($S/code/pipeline.sh recipe).
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_metric
S=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/seq_invisible_20261111
N=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/native_run_20260910
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
QFS=$N/query/querydb_foldseek
TFS=$N/reference/keggclusterdb_foldseek
T="${SLURM_CPUS_ON_NODE:?}"
TAG="${TAG:-full}"
D="$W/$TAG/tm"
export OMP_NUM_THREADS="$T" PYTHONDONTWRITEBYTECODE=1
cd "$W/code"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T tag=$TAG"
"$PY" -u tm_sample.py prep --threads "$T" --tag "$TAG"
rm -rf "$D/db"; mkdir -p "$D/db/tref"
ln -s "$TFS" "$D/db/tref/tref"; ln -s "$TFS.index" "$D/db/tref/tref.index"; cp "$TFS.dbtype" "$D/db/tref/tref.dbtype"
for s in ca ss; do
  ln -s "${TFS}_$s" "$D/db/tref/tref_$s"; ln -s "${TFS}_$s.index" "$D/db/tref/tref_$s.index"; cp "${TFS}_$s.dbtype" "$D/db/tref/tref_$s.dbtype"
done
"$MM" tsv2db "$S/work/tref_h.tsv" "$D/db/tref/tref_h" --output-dbtype 12 -v 2
"$MM" tsv2db "$D/pref_tm.tsv" "$D/db/pref" --output-dbtype 7 -v 2
stamp "START tmalign"
"$FS" tmalign "$QFS" "$TFS" "$D/db/pref" "$D/db/aln_tm" --tmscore-threshold 0.0 -a 1 --threads "$T" -v 2
"$FS" convertalis "$QFS" "$D/db/tref/tref" "$D/db/aln_tm" "$D/aln_tm.tsv" \
    --format-output query,target,qtmscore,ttmscore,alntmscore,lddt,fident,qcov,tcov,alnlen,qlen,tlen,evalue,bits \
    --exact-tmscore 1 --threads "$T" -v 2
stamp "DONE tmalign"
wc -l "$D/pref_tm.tsv" "$D/aln_tm.tsv"
"$PY" -u tm_sample.py summarize --threads "$T" --tag "$TAG"
stamp "DONE"
"$PY" -u report_tables.py --tag "$TAG"
