#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sa_compare
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors/logs/compare_%j.log

# Decisive comparison, structure sample, foldseek TM-align (exact TM-score) on pairs not scored tonight.
set -euo pipefail
ROOT=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust
N=$ROOT/native_run_20260910
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
T="${SLURM_CPUS_ON_NODE:?}"
QFS=$N/query/querydb_foldseek
TFS=$N/reference/keggclusterdb_foldseek
K=$W/work/tm
D=$K/db
export PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T"

stamp "START prep"
"$PY" -u "$W/code/compare.py" prep --threads "$T"
stamp "DONE prep"
rm -rf "$D"; mkdir -p "$D/q" "$D/t"
# aliased structure DBs whose headers carry K<key> (read results by key; trap 6)
for pair in "q:$QFS" "t:$TFS"; do
    s=${pair%%:*}; src=${pair#*:}
    ln -s "$src" "$D/$s/$s"; ln -s "$src.index" "$D/$s/$s.index"; cp "$src.dbtype" "$D/$s/$s.dbtype"
    for x in ca ss; do
        ln -s "${src}_$x" "$D/$s/${s}_$x"; ln -s "${src}_$x.index" "$D/$s/${s}_$x.index"; cp "${src}_$x.dbtype" "$D/$s/${s}_$x.dbtype"
    done
    "$MM" tsv2db "$K/${s}_h.tsv" "$D/$s/${s}_h" --output-dbtype 12 -v 2
done
n=$(wc -l < "$K/pref.tsv")
stamp "pairs to align: $n"
if [ "$n" -gt 0 ]; then
    "$MM" tsv2db "$K/pref.tsv" "$D/pref" --output-dbtype 7 -v 2
    stamp "START foldseek_tmalign"
    "$FS" tmalign "$D/q/q" "$D/t/t" "$D/pref" "$D/aln_tm" --tmscore-threshold 0.0 -a 1 --threads "$T" -v 2
    stamp "DONE foldseek_tmalign"
    "$FS" convertalis "$D/q/q" "$D/t/t" "$D/aln_tm" "$K/aln_tm.tsv" \
        --format-output query,target,qtmscore,ttmscore,alntmscore,lddt,fident,qcov,tcov,alnlen,qlen,tlen,evalue,bits \
        --exact-tmscore 1 --threads "$T" -v 2
    stamp "DONE tm_convertalis"
    wc -l "$K/aln_tm.tsv"
fi
"$PY" -u "$W/code/compare.py" summarize --threads "$T"
stamp "DONE summarize"
