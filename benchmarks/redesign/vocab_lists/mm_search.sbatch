#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J vl_mmseqs
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/vocab_lists/logs/mm_search_%j.log
# ws_vocab2b alignment-list control: MMseqs2 18 all-against-all of the 2,547,528 unique sequences
# (prefilter -s 7.5 --max-seqs 1000; align -e 1e-3; no coverage rule), then per query the hits ranked by bit score
# -> $RD/vocab_lists/mm/aln__k1000__{I,B}.npy (the kNN array layout used by softrec's list expansion).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
U=$RD/subgenic/db/uniqdb
X=$RD/vocab_lists/mm
mkdir -p "$X/tmp"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T"
"$MM" rmdb "$X/all_pref" -v 1 2>/dev/null || true
"$MM" rmdb "$X/all_aln" -v 1 2>/dev/null || true
t0=$(date +%s); stamp "START prefilter"
"$MM" prefilter "$U" "$U" "$X/all_pref" -s 7.5 --max-seqs 1000 --threads "$T" -v 3 > "$X/prefilter.log" 2>&1
t1=$(date +%s); stamp "START align (prefilter $((t1 - t0)) s)"
"$MM" align "$U" "$U" "$X/all_pref" "$X/all_aln" -e 1e-3 --threads "$T" -v 3 > "$X/align.log" 2>&1
t2=$(date +%s); stamp "DONE align ($((t2 - t1)) s)"
"$MM" rmdb "$X/all_pref" -v 2
"$MM" convertalis "$U" "$U" "$X/all_aln" "$X/all.tsv" --format-output query,target,bits --threads "$T" -v 2
t3=$(date +%s); stamp "DONE convertalis ($((t3 - t2)) s)"
OMP_NUM_THREADS=$T "$PY" -u "$RD/vocab_lists/code/mm2knn.py" "$X/all.tsv" "$X" 1000
t4=$(date +%s); stamp "DONE arrays ($((t4 - t3)) s)"
echo -e "prefilter\t$((t1 - t0))\talign\t$((t2 - t1))\tconvert\t$((t3 - t2))\tarrays\t$((t4 - t3))" > "$X/timing.tsv"
rm -f "$X/all.tsv"
stamp "done"
