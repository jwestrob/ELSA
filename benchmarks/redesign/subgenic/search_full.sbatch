#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sg_search
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/subgenic_search_%j.log
# Sub-genic sizing, job 2 (sampled design): S = non-giant unique sequences of a random genome sample plus
# gold neighbourhoods vs all unique sequences, local, no coverage filter (filtered later); B = giants vs all,
# local; C = giants vs giants; D = giant self-repeats via segments. Analysis runs as its own job.
# Settings come from the environment (sbatch --export=ALL): NGEN, SA, MA (search S), MB (search B), STAGES.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
W=$RD/subgenic; C=$W/code; D=$W/db; H=$W/hits; X=$W/run
NGEN=${NGEN:?}; SA=${SA:?}; MA=${MA:?}; MB=${MB:?}; STAGES=${STAGES:-S,B,C,D}
mkdir -p "$H" "$X" "$W/rep" "$W/tmp"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T NGEN=$NGEN SA=$SA MA=$MA MB=$MB STAGES=$STAGES"; "$MM" version
has() { [[ ",$STAGES," == *",$1,"* ]]; }
secs() { echo $(( $(date +%s) - $1 )); }
TIMING=$W/run/timing_$SLURM_JOB_ID.tsv

if has S; then
  "$PY" -u "$C/make_queries.py" --work "$W" --pkg "$P" --n-genomes "$NGEN"
  rm -rf "$D/S" "$D/S".*; "$MM" createsubdb "$W/lists/S.uid" "$D/uniqdb" "$D/S" -v 2
fi
for L in giants; do
  [ -s "$D/$L.dbtype" ] || "$MM" createsubdb "$W/lists/$L.uid" "$D/uniqdb" "$D/$L" -v 2
done

conv () {  # qdb tdb aln out.parquet
  "$MM" convertalis "$1" "$2" "$3" "$X/conv.tsv" \
     --format-output query,target,evalue,bits,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
  "$PY" -u "$C/tsv2parquet.py" "$X/conv.tsv" "$4"
  rm -f "$X/conv.tsv"
}

if has D; then
  stamp "START D (internal repeats)"; t0=$(date +%s)
  RP=$W/rep; rm -rf "$RP"/db; mkdir -p "$RP/db"
  "$PY" -u "$C/make_segments.py" --work "$W"
  "$MM" createdb "$RP/segments.faa" "$RP/db/seg" --shuffle 0 -v 2
  "$MM" createdb "$RP/parents.faa" "$RP/db/par" --shuffle 0 -v 2
  for f in seg par; do
    awk -F'\t' -v f=$f 'BEGIN{bad=0} {if ($1 != NR-1) bad++} END{print f, "lookup rows", NR, "out-of-order keys", bad; exit (bad>0)}' "$RP/db/$f.lookup"
  done
  "$MM" tsv2db "$RP/pref.tsv" "$RP/db/pref" --output-dbtype 7 -v 2
  "$MM" align "$RP/db/seg" "$RP/db/par" "$RP/db/pref" "$RP/db/aln" -e 1e-3 --alt-ali 50 --alignment-mode 3 -a --threads "$T" -v 2
  "$MM" convertalis "$RP/db/seg" "$RP/db/par" "$RP/db/aln" "$RP/aln.tsv" \
     --format-output query,target,evalue,bits,qstart,qend,qlen,tstart,tend,tlen,alnlen,pident --threads "$T" -v 2
  wc -l "$RP/aln.tsv"; echo -e "D_total\t$(secs $t0)" >> "$TIMING"
  stamp "DONE D"
fi

if has C; then
  stamp "START C (giants vs giants)"; t0=$(date +%s)
  "$MM" prefilter "$D/giants" "$D/giants" "$X/C_pref" -s 7.5 --max-seqs 5000 --threads "$T" -v 2
  "$MM" align "$D/giants" "$D/giants" "$X/C_pref" "$X/C_aln" -e 1e-5 -a --threads "$T" -v 2
  "$MM" rmdb "$X/C_pref" -v 2
  conv "$D/giants" "$D/giants" "$X/C_aln" "$H/C.parquet"; echo -e "C_total\t$(secs $t0)" >> "$TIMING"
  stamp "DONE C"
fi

if has B; then
  stamp "START B (giants vs all, local) max-seqs=$MB"; t0=$(date +%s)
  "$MM" prefilter "$D/giants" "$D/uniqdb" "$X/B_pref" -s 7.5 --max-seqs "$MB" --threads "$T" -v 3 > "$X/B_prefilter.log" 2>&1
  "$MM" align "$D/giants" "$D/uniqdb" "$X/B_pref" "$X/B_aln" -e 1e-5 -a --threads "$T" -v 3 > "$X/B_align.log" 2>&1
  "$MM" result2stats "$D/giants" "$D/uniqdb" "$X/B_pref" "$X/B_pref_lc" --stat linecount --threads "$T" -v 2
  "$MM" prefixid "$X/B_pref_lc" "$X/B_pref_lc.tsv" --tsv --threads "$T" -v 2
  "$MM" rmdb "$X/B_pref" -v 2
  conv "$D/giants" "$D/uniqdb" "$X/B_aln" "$H/B.parquet"; echo -e "B_total\t$(secs $t0)" >> "$TIMING"
  stamp "DONE B"
fi

if has S; then
  stamp "START S prefilter s=$SA max-seqs=$MA"; t0=$(date +%s)
  "$MM" prefilter "$D/S" "$D/uniqdb" "$X/S_pref" -s "$SA" --max-seqs "$MA" --threads "$T" -v 3 > "$X/S_prefilter.log" 2>&1
  echo -e "S_prefilter\t$(secs $t0)" >> "$TIMING"; stamp "START S align"; t0=$(date +%s)
  "$MM" align "$D/S" "$D/uniqdb" "$X/S_pref" "$X/S_aln" -e 1e-5 -a --threads "$T" -v 3 > "$X/S_align.log" 2>&1
  echo -e "S_align\t$(secs $t0)" >> "$TIMING"
  "$MM" result2stats "$D/S" "$D/uniqdb" "$X/S_pref" "$X/S_pref_lc" --stat linecount --threads "$T" -v 2
  "$MM" prefixid "$X/S_pref_lc" "$X/S_pref_lc.tsv" --tsv --threads "$T" -v 2
  "$MM" rmdb "$X/S_pref" -v 2
  t0=$(date +%s); conv "$D/S" "$D/uniqdb" "$X/S_aln" "$H/S.parquet"; echo -e "S_convert\t$(secs $t0)" >> "$TIMING"
  stamp "DONE S"
fi

cat "$TIMING"
