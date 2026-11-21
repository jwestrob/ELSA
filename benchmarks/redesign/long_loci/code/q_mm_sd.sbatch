#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ll_mmsd
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_mmsd_%j.log
# long_loci part 3: MMseqs2 18 all-against-all of the Spacedust collection's 4,231,484 unique sequences with the DPANN
# control's parameters (vocab_lists mm_search.sbatch: prefilter -s 7.5 --max-seqs 1000; align -e 1e-3; no coverage
# rule), then per query the hits ranked by bit score -> $LL/mm_sd/aln__k1000__{I,B}.npy (uid rows = fam/gid2uid uids).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
X=$LL/mm_sd
mkdir -p "$X/tmp"
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
if [[ ! -f "$X/uniqdb.dbtype" ]]; then
  "$MM" createdb "$TS/fam/uniq.faa" "$X/uniqdb" --dbtype 1 -v 2 > "$X/createdb.log" 2>&1
fi
N=$(grep -c "^>" "$TS/fam/uniq.faa")
stamp "uniqdb ready: $N sequences"
"$MM" rmdb "$X/all_pref" -v 1 2>/dev/null || true
"$MM" rmdb "$X/all_aln" -v 1 2>/dev/null || true
t0=$(date +%s); stamp "START prefilter"
"$MM" prefilter "$X/uniqdb" "$X/uniqdb" "$X/all_pref" -s 7.5 --max-seqs 1000 --threads "$T" -v 3 > "$X/prefilter.log" 2>&1
t1=$(date +%s); stamp "START align (prefilter $((t1 - t0)) s)"
"$MM" align "$X/uniqdb" "$X/uniqdb" "$X/all_pref" "$X/all_aln" -e 1e-3 --threads "$T" -v 3 > "$X/align.log" 2>&1
t2=$(date +%s); stamp "DONE align ($((t2 - t1)) s)"
"$MM" rmdb "$X/all_pref" -v 2
"$MM" convertalis "$X/uniqdb" "$X/uniqdb" "$X/all_aln" "$X/all.tsv" --format-output query,target,bits --threads "$T" -v 2
t3=$(date +%s); stamp "DONE convertalis ($((t3 - t2)) s)"
OMP_NUM_THREADS=$T "$PYT" -u "$C/mm2knn_sd.py" "$X/all.tsv" "$X" 1000 "$N"
t4=$(date +%s); stamp "DONE arrays ($((t4 - t3)) s)"
echo -e "prefilter\t$((t1 - t0))\talign\t$((t2 - t1))\tconvert\t$((t3 - t2))\tarrays\t$((t4 - t3))" > "$X/timing.tsv"
rm -f "$X/all.tsv"
stamp "ALL DONE"
