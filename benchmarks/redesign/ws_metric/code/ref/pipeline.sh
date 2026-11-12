#!/bin/bash
# Sequence-invisible / structure-confirmed pair analysis.
# usage: pipeline.sh RUN_DIR THREADS [SMOKE_BLOCKS] [MODELS]
set -euo pipefail

RUN=$1
T=$2
SMOKE=${3:-0}
MODELS=${4:-esm2_8m,esm2_35m,esm2_150m,esm2_650m,esmc300m,esmc600m,esmc6b}

R=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust
N=$R/native_run_20260910
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
CODE=$(dirname "$(readlink -f "$0")")

QAA=$N/query/querydb
TAA=$N/reference/keggclusterdb
QFS=$N/query/querydb_foldseek
TFS=$N/reference/keggclusterdb_foldseek

W=$RUN/work
D=$W/db
mkdir -p "$W" "$RUN/results" "$RUN/tmp"
rm -rf "$D"
mkdir -p "$D/tref"

stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T smoke=$SMOKE models=$MODELS"
"$MM" version
"$FS" version

stamp "START prep"
"$PY" "$CODE/prep_pairs.py" --work "$RUN" --threads "$T" --smoke "$SMOKE" --models "$MODELS"
stamp "DONE prep"
wc -l "$W/pref_all.tsv" "$W/pref_struct.tsv"

# ---- sequence: Smith-Waterman on every pair -------------------------------
stamp "START mmseqs_align"
"$MM" tsv2db "$W/pref_all.tsv" "$D/pref_all" --output-dbtype 7 -v 2
"$MM" align "$QAA" "$TAA" "$D/pref_all" "$D/aln_seq" -e inf --alignment-mode 3 --threads "$T" -v 2
"$MM" prefixid "$D/aln_seq" "$W/aln_seq.tsv" --tsv --threads "$T" -v 2
stamp "DONE mmseqs_align"
wc -l "$W/aln_seq.tsv"

# ---- structure: aliased target DB whose headers carry the numeric key ------
ln -s "$TFS" "$D/tref/tref"
ln -s "$TFS.index" "$D/tref/tref.index"
cp "$TFS.dbtype" "$D/tref/tref.dbtype"
for s in ca ss; do
    ln -s "${TFS}_$s" "$D/tref/tref_$s"
    ln -s "${TFS}_$s.index" "$D/tref/tref_$s.index"
    cp "${TFS}_$s.dbtype" "$D/tref/tref_$s.dbtype"
done
"$MM" tsv2db "$W/tref_h.tsv" "$D/tref/tref_h" --output-dbtype 12 -v 2
"$MM" tsv2db "$W/pref_struct.tsv" "$D/pref_struct" --output-dbtype 7 -v 2

stamp "START foldseek_tmalign"
"$FS" tmalign "$QFS" "$TFS" "$D/pref_struct" "$D/aln_tm" --tmscore-threshold 0.0 -a 1 --threads "$T" -v 2
stamp "DONE foldseek_tmalign"
"$MM" prefixid "$D/aln_tm" "$W/aln_tm_raw.tsv" --tsv --threads "$T" -v 2
stamp "START tm_convertalis"
"$FS" convertalis "$QFS" "$D/tref/tref" "$D/aln_tm" "$W/aln_tm.tsv" \
    --format-output query,target,qtmscore,ttmscore,alntmscore,lddt,fident,qcov,tcov,alnlen,qlen,tlen,evalue,bits \
    --exact-tmscore 1 --threads "$T" -v 2
stamp "DONE tm_convertalis"
wc -l "$W/aln_tm_raw.tsv" "$W/aln_tm.tsv"

stamp "START foldseek_3di"
if "$FS" structurealign "$QFS" "$TFS" "$D/pref_struct" "$D/aln_3di" -e 10 -a 1 --threads "$T" -v 2 \
   && "$FS" convertalis "$QFS" "$D/tref/tref" "$D/aln_3di" "$W/aln_3di.tsv" \
        --format-output query,target,evalue,bits,prob,fident,qcov,tcov,alntmscore,qtmscore,ttmscore,lddt \
        --threads "$T" -v 2; then
    stamp "DONE foldseek_3di"
    wc -l "$W/aln_3di.tsv"
else
    stamp "FAILED foldseek_3di (non-fatal; summary proceeds on TM-align)"
    rm -f "$W/aln_3di.tsv"
fi

stamp "START summarize"
"$PY" "$CODE/summarize.py" --run "$RUN" --threads "$T"
stamp "DONE summarize"
