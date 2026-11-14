#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J tok_mmseqs
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/mmseqs_%j.log

# Arm mmseqs__seq__id30c80: MMseqs2 clustering of all 4,257,531 universe proteins.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
WS=$RD/ws_tokens
W=$WS/work/mmseqs
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
T="${SLURM_CPUS_ON_NODE:?}"
stamp() { printf '%s %s\n' "$(date -Is)" "$*"; }
cd "$WS/code"
mkdir -p "$W"
stamp "host=$(hostname) threads=$T mmseqs=$($MM version)"

if [ ! -s "$W/all.faa" ]; then
    stamp "START fasta"
    "$PY" -u make_fasta.py --out "$W/all.faa.tmp"
    mv "$W/all.faa.tmp" "$W/all.faa"; mv "$W/all.faa.tmp.stats.json" "$W/all.faa.stats.json"
fi
grep -c '^>' "$W/all.faa"

if [ ! -s "$W/seqdb.index" ]; then
    stamp "START createdb"
    "$MM" createdb "$W/all.faa" "$W/seqdb" --dbtype 1 --shuffle 0 -v 2
fi
wc -l "$W/seqdb.index" "$W/seqdb.lookup"

stamp "START cluster"
rm -rf "$W/tmp" "$W"/clu.* "$W"/clu
t0=$(date +%s)
/usr/bin/time -v -o "$W/cluster.time" \
    "$MM" cluster "$W/seqdb" "$W/clu" "$W/tmp" --min-seq-id 0.3 -c 0.8 --cov-mode 0 --threads "$T" -v 3 \
    > "$W/cluster.log" 2>&1
t1=$(date +%s)
stamp "DONE cluster $((t1 - t0)) s"
tail -3 "$W/cluster.log"
"$MM" createtsv "$W/seqdb" "$W/seqdb" "$W/clu" "$W/clu.tsv" --threads "$T" -v 2
wc -l "$W/clu.tsv"
rss=$(awk -F': ' '/Maximum resident set size/ {print $2}' "$W/cluster.time")
"$PY" -u mmseqs_tokens.py --work "$W" --seconds-cluster $((t1 - t0)) --peak-rss-kb "$rss" \
    --mmseqs "$MM" --mmseqs-version "$($MM version)" --threads "$T"
rm -rf "$W/tmp"
stamp "END"
