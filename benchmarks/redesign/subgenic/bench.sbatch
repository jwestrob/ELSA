#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sg_bench
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/subgenic_bench_%j.log
# Sub-genic sizing, job 1: unique sequences, MMseqs DB, and a timing/sensitivity benchmark of the
# target-coverage search on 20,000 random non-giant unique queries (+ 200 giants, local search).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
W=$RD/subgenic
C=$W/code
D=$W/db
B=$W/bench
mkdir -p "$D" "$B" "$W/tmp"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T"; "$MM" version; lscpu | grep -E "Model name|Flags" | cut -c1-200

if [ ! -s "$W/uniq.parquet" ]; then
  stamp "START prep_unique"
  "$PY" -u "$C/prep_unique.py" --faa "$RD/ws_dpann/work/proteins_gid.faa" --genes "$P/universe/genes.parquet" --out "$W"
  stamp "DONE prep_unique"
fi
if [ ! -s "$D/uniqdb.dbtype" ]; then
  "$MM" createdb "$W/uniq.faa" "$D/uniqdb" --shuffle 0 -v 2
fi
# key must equal uid (header) for the uid lists to work as createsubdb key lists
awk -F'\t' '$1!=$2{bad++} END{print "lookup key!=header rows:", bad+0, "of", NR; exit (bad>0)}' "$D/uniqdb.lookup"

for L in bench20k bench_giant200; do
  [ -s "$D/$L.dbtype" ] || "$MM" createsubdb "$W/lists/$L.uid" "$D/uniqdb" "$D/$L" -v 2
done

run_cfg () {  # name query s maxseqs covflags
  local N=$1 Q=$2 S=$3 M=$4; shift 4
  local O=$B/$N; rm -rf "$O"; mkdir -p "$O"
  stamp "START $N prefilter"
  local t0=$(date +%s)
  "$MM" prefilter "$D/$Q" "$D/uniqdb" "$O/pref" -s "$S" --max-seqs "$M" "$@" --threads "$T" -v 3 > "$O/prefilter.log" 2>&1
  local t1=$(date +%s)
  stamp "START $N align"
  "$MM" align "$D/$Q" "$D/uniqdb" "$O/pref" "$O/aln" -e 1e-5 -a "$@" --threads "$T" -v 3 > "$O/align.log" 2>&1
  local t2=$(date +%s)
  "$MM" result2stats "$D/$Q" "$D/uniqdb" "$O/pref" "$O/pref_lc" --stat linecount --threads "$T" -v 2
  "$MM" prefixid "$O/pref_lc" "$O/pref_lc.tsv" --tsv --threads "$T" -v 2
  "$MM" convertalis "$D/$Q" "$D/uniqdb" "$O/aln" "$O/aln.tsv" \
     --format-output query,target,evalue,bits,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
  local t3=$(date +%s)
  echo "{\"name\":\"$N\",\"query\":\"$Q\",\"s\":$S,\"max_seqs\":$M,\"extra\":\"$*\",\"prefilter_s\":$((t1-t0)),\"align_s\":$((t2-t1)),\"convert_s\":$((t3-t2)),\"threads\":$T}" > "$O/timing.json"
  grep -E "k-mer|Time for|Index table|kmer|Sensitivity|Query db start|splits|Estimated" "$O/prefilter.log" | head -20 || true
  cat "$O/timing.json"
  rm -f "$O"/pref.[0-9]* "$O"/aln.[0-9]*
}

COV="--cov-mode 1 -c 0.7"
run_cfg s75_m4000 bench20k 7.5 4000 $COV
run_cfg s57_m4000 bench20k 5.7 4000 $COV
run_cfg s57_m1000 bench20k 5.7 1000 $COV
run_cfg s75_m1000 bench20k 7.5 1000 $COV
run_cfg giant_s75_m20000 bench_giant200 7.5 20000

stamp "START bench_summary"
"$PY" -u "$C/bench_summary.py" --work "$W"
stamp "DONE"
