#!/bin/bash
# MMseqs2 Smith-Waterman on an explicit pair list (recipe of seq_invisible_20261111/code/pipeline.sh).
# usage: align_pairs.sh PREF_TSV OUT_DIR THREADS     (PREF_TSV: qkey tkey 0 0, sorted by qkey, tkey)
set -euo pipefail
PREF=$1; OUT=$2; T=$3
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
N=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/native_run_20260910
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
"$MM" version
rm -rf "$OUT/db"; mkdir -p "$OUT/db"
stamp "START align $(wc -l < "$PREF") pairs"
"$MM" tsv2db "$PREF" "$OUT/db/pref" --output-dbtype 7 -v 2
"$MM" align "$N/query/querydb" "$N/reference/keggclusterdb" "$OUT/db/pref" "$OUT/db/aln" -e inf --alignment-mode 3 --threads "$T" -v 2
"$MM" prefixid "$OUT/db/aln" "$OUT/aln.tsv.tmp" --tsv --threads "$T" -v 2
mv "$OUT/aln.tsv.tmp" "$OUT/aln.tsv"
stamp "DONE align $(wc -l < "$OUT/aln.tsv") alignments"
rm -rf "$OUT/db"
