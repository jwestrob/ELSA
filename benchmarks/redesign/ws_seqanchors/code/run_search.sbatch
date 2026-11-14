#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sa_search
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors/logs/search_%j.log

# MMseqs2 release 18: 67,148 query proteins vs 4,190,383 reference proteins,
# sensitivity 7.5, E <= 1e-3, 4,000 prefilter candidates per query.
set -euo pipefail
ROOT=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust
N=$ROOT/native_run_20260910
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
MAXSEQS=4000
D=$W/work/search
rm -rf "$D"; mkdir -p "$D"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T maxseqs=$MAXSEQS"
"$MM" version
lscpu | grep -E "Model name|Flags" | cut -c1-200

t0=$(date +%s)
stamp "START prefilter"
"$MM" prefilter "$N/query/querydb" "$N/reference/keggclusterdb" "$D/pref" \
    -s 7.5 --max-seqs "$MAXSEQS" --threads "$T" -v 2
t1=$(date +%s); stamp "DONE prefilter $((t1-t0))s"
stamp "START align"
"$MM" align "$N/query/querydb" "$N/reference/keggclusterdb" "$D/pref" "$D/aln" \
    -e 1e-3 --alignment-mode 3 --threads "$T" -v 2
t2=$(date +%s); stamp "DONE align $((t2-t1))s"
"$MM" prefixid "$D/aln" "$W/work/hits.tsv" --tsv --threads "$T" -v 2
"$MM" result2stats "$N/query/querydb" "$N/reference/keggclusterdb" "$D/pref" "$D/pref_count" \
    --stat linecount --threads "$T" -v 2
"$MM" prefixid "$D/pref_count" "$W/work/pref_count.tsv" --tsv --threads "$T" -v 2
t3=$(date +%s); stamp "DONE export $((t3-t2))s"
wc -l "$W/work/hits.tsv" "$W/work/pref_count.tsv"
printf '{"prefilter_seconds": %d, "align_seconds": %d, "export_seconds": %d, "search_seconds": %d, "threads": %d, "host": "%s", "max_seqs": %d, "sensitivity": 7.5, "evalue": 0.001, "mmseqs": "%s"}\n' \
    $((t1-t0)) $((t2-t1)) $((t3-t2)) $((t2-t0)) "$T" "$(hostname)" "$MAXSEQS" "$("$MM" version)" > "$W/work/search_timing.json"
stamp "START post"
"$PY" -u "$W/code/search_post.py" --threads "$T" --max-seqs "$MAXSEQS" --timing "$W/work/search_timing.json"
stamp "DONE post"
