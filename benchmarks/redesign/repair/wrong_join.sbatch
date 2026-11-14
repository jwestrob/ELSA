#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J rp_wjoin
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/repair_wrong_join_%j.log
# Cut test diagnostic: rescore (saving the joined target), then align each intact original to the target its fragment
# joined (explicit pairs, -e 1e-5 -a) and classify the wrong joins.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
R=$RD/repair; C=$R/code; X=$R/cut; U=$RD/subgenic/db/uniqdb
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PY" -u "$C/cut_test_score.py" "$R" "$RD/subgenic" > "$X/score_stdout.txt"
"$PY" -u "$C/wrong_join_check.py" prep "$R"
for f in orig_pref orig_aln; do "$MM" rmdb "$X/$f" -v 1 2>/dev/null || true; done
"$MM" tsv2db "$X/orig_pref.tsv" "$X/orig_pref" --output-dbtype 7 -v 2
"$MM" align "$U" "$U" "$X/orig_pref" "$X/orig_aln" -e 1e-5 -a --alignment-mode 3 --threads "$T" -v 2
"$MM" convertalis "$U" "$U" "$X/orig_aln" "$X/orig_hits.tsv" \
   --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
"$PY" -u "$C/tsv2parquet.py" "$X/orig_hits.tsv" "$X/orig_hits.parquet" keep_all
"$PY" -u "$C/wrong_join_check.py" classify "$R"
