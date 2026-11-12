#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J rp_search
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/repair_search_%j.log
# Workstream repair, one query shard: lists/Q_K.uid (rescue + split-short queries) against all unique sequences.
# prefilter -s 7.5 --max-seqs 1000; align -e 1e-5 -a; no coverage filter (applied in code). Environment: K (exported).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
R=$RD/repair; C=$R/code; D=$R/db; H=$R/hits; X=$R/run; U=$RD/subgenic/db/uniqdb
K=${K:?}; SA=7.5; MA=1000
mkdir -p "$D" "$H" "$X" "$R/tmp"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T shard=$K s=$SA max-seqs=$MA"
wc -l "$R/lists/Q_$K.uid"
"$MM" rmdb "$D/Q_$K" -v 1 2>/dev/null || true
"$MM" createsubdb "$R/lists/Q_$K.uid" "$U" "$D/Q_$K" -v 2
stamp "START prefilter"; t0=$(date +%s)
"$MM" prefilter "$D/Q_$K" "$U" "$X/Q_${K}_pref" -s "$SA" --max-seqs "$MA" --threads "$T" -v 3 > "$X/Q_${K}_prefilter.log" 2>&1
t1=$(date +%s); stamp "START align (prefilter $((t1 - t0)) s)"
"$MM" align "$D/Q_$K" "$U" "$X/Q_${K}_pref" "$X/Q_${K}_aln" -e 1e-5 -a --threads "$T" -v 3 > "$X/Q_${K}_align.log" 2>&1
t2=$(date +%s); stamp "DONE align ($((t2 - t1)) s)"
"$MM" result2stats "$D/Q_$K" "$U" "$X/Q_${K}_pref" "$X/Q_${K}_pref_lc" --stat linecount --threads "$T" -v 2
"$MM" prefixid "$X/Q_${K}_pref_lc" "$X/Q_${K}_pref_lc.tsv" --tsv --threads "$T" -v 2
"$MM" rmdb "$X/Q_${K}_pref" -v 2
"$MM" convertalis "$D/Q_$K" "$U" "$X/Q_${K}_aln" "$X/conv_Q_$K.tsv" \
   --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
"$PY" -u "$C/tsv2parquet.py" "$X/conv_Q_$K.tsv" "$H/Q_$K.parquet"
rm -f "$X/conv_Q_$K.tsv"
t3=$(date +%s)
echo -e "Q_${K}\tprefilter\t$((t1 - t0))\talign\t$((t2 - t1))\tconvert\t$((t3 - t2))" >> "$R/run/timing_shards.tsv"
stamp "DONE shard $K"
