#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ts_fam
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/logs/ts_fam_%j.log
# transfer_spacedust: rescue3 family tokens for the Spacedust collection (DPANN recipe; repair workstream's search
# parameters: prefilter -s 7.5 --max-seqs 1000, align -e 1e-5 -a, coverage applied in code).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
TS=$RD/transfer_spacedust; C=$TS/code; F=$TS/fam
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T NUMBA_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
mkdir -p "$F/db" "$F/run" "$F/hits" "$F/tmp"
/usr/bin/time -v "$PY" -u "$C/fam.py" uniq --out "$F" 2> "$F/run/uniq.time"
stamp "uniq done"
"$MM" createdb "$F/uniq.faa" "$F/db/uniqdb" --shuffle 0 -v 2
# keys must equal uids (the header) for createsubdb on uid lists
awk '$1 != $2 {bad++} END {print "lookup rows", NR, "key!=uid", bad+0; if (bad) exit 1}' "$F/db/uniqdb.lookup"
"$MM" createsubdb "$F/Q.uid" "$F/db/uniqdb" "$F/db/Q" -v 2
stamp "START prefilter"; t0=$(date +%s)
"$MM" prefilter "$F/db/Q" "$F/db/uniqdb" "$F/run/pref" -s 7.5 --max-seqs 1000 --threads "$T" -v 3 > "$F/run/prefilter.log" 2>&1
t1=$(date +%s); stamp "START align (prefilter $((t1 - t0)) s)"
"$MM" align "$F/db/Q" "$F/db/uniqdb" "$F/run/pref" "$F/run/aln" -e 1e-5 -a --threads "$T" -v 3 > "$F/run/align.log" 2>&1
t2=$(date +%s); stamp "DONE align ($((t2 - t1)) s)"
"$MM" result2stats "$F/db/Q" "$F/db/uniqdb" "$F/run/pref" "$F/run/pref_lc" --stat linecount --threads "$T" -v 2
"$MM" prefixid "$F/run/pref_lc" "$F/run/pref_lc.tsv" --tsv --threads "$T" -v 2
"$MM" rmdb "$F/run/pref" -v 2
"$MM" convertalis "$F/db/Q" "$F/db/uniqdb" "$F/run/aln" "$F/run/conv.tsv" \
   --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
"$PY" -u "$C/tsv2parquet.py" "$F/run/conv.tsv" "$F/hits/Q_0.parquet"
rm -f "$F/run/conv.tsv"
t3=$(date +%s)
echo -e "prefilter\t$((t1 - t0))\talign\t$((t2 - t1))\tconvert\t$((t3 - t2))" > "$F/run/timing.tsv"
awk '{n++; if ($2 >= 1000) full++} END {print "prefilter result lines per query: queries", n, "at max-seqs 1000:", full+0}' "$F/run/pref_lc.tsv"
/usr/bin/time -v "$PY" -u "$C/fam.py" rescue3 --dir "$F" --threads "$T" 2> "$F/run/rescue3.time"
stamp "DONE"
