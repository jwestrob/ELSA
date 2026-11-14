#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sg_shard
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/subgenic_shard_%j.log
# Sub-genic sizing, search S as query shards: shard K of N takes every N-th line of lists/S.uid
# (round-robin over uid order), same settings as the unsharded search: prefilter -s SA --max-seqs MA,
# align -e 1e-5 -a, no coverage filter. Environment: K, N, SA, MA (exported, sbatch --export=ALL).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
W=$RD/subgenic; C=$W/code; D=$W/db; H=$W/hits; X=$W/run
K=${K:?}; N=${N:?}; SA=${SA:?}; MA=${MA:?}
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T shard=$K/$N s=$SA max-seqs=$MA"
awk -v n="$N" -v k="$K" '(NR - 1) % n == k' "$W/lists/S.uid" > "$W/lists/S_$K.uid"
wc -l "$W/lists/S_$K.uid"
"$MM" rmdb "$D/S_$K" -v 1 2>/dev/null || true
"$MM" createsubdb "$W/lists/S_$K.uid" "$D/uniqdb" "$D/S_$K" -v 2
stamp "START prefilter"; t0=$(date +%s)
"$MM" prefilter "$D/S_$K" "$D/uniqdb" "$X/S_${K}_pref" -s "$SA" --max-seqs "$MA" --threads "$T" -v 3 > "$X/S_${K}_prefilter.log" 2>&1
t1=$(date +%s); stamp "START align (prefilter $((t1 - t0)) s)"
"$MM" align "$D/S_$K" "$D/uniqdb" "$X/S_${K}_pref" "$X/S_${K}_aln" -e 1e-5 -a --threads "$T" -v 3 > "$X/S_${K}_align.log" 2>&1
t2=$(date +%s); stamp "DONE align ($((t2 - t1)) s)"
"$MM" result2stats "$D/S_$K" "$D/uniqdb" "$X/S_${K}_pref" "$X/S_${K}_pref_lc" --stat linecount --threads "$T" -v 2
"$MM" prefixid "$X/S_${K}_pref_lc" "$X/S_${K}_pref_lc.tsv" --tsv --threads "$T" -v 2
"$MM" rmdb "$X/S_${K}_pref" -v 2
"$MM" convertalis "$D/S_$K" "$D/uniqdb" "$X/S_${K}_aln" "$X/conv_S_$K.tsv" \
   --format-output query,target,evalue,bits,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
"$PY" -u "$C/tsv2parquet.py" "$X/conv_S_$K.tsv" "$H/S_$K.parquet"
rm -f "$X/conv_S_$K.tsv"
t3=$(date +%s)
echo -e "S_${K}\tprefilter\t$((t1 - t0))\talign\t$((t2 - t1))\tconvert\t$((t3 - t2))" >> "$W/run/timing_shards.tsv"
stamp "DONE shard $K"
