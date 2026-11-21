#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J rp_rerr
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/repair_rescue_error_%j.log
# Workstream repair, rescue error (predeclared amendment_1): truncation flags, cut-test fragments, their search with
# the genome-wide settings (-s 7.5 --max-seqs 1000; align -e 1e-5 -a), cut-test scoring, genome-wide variants a-d.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
R=$RD/repair; C=$R/code; X=$R/cut; U=$RD/subgenic/db/uniqdb
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T"
"$PY" -u "$C/cut_test_make.py" "$R" "$RD/subgenic" "$RD/ws_dpann/package"
for f in fragdb fragpref fragaln; do "$MM" rmdb "$X/$f" -v 1 2>/dev/null || true; done
"$MM" createdb "$X/frag.faa" "$X/fragdb" -v 2
stamp "START prefilter"
"$MM" prefilter "$X/fragdb" "$U" "$X/fragpref" -s 7.5 --max-seqs 1000 --threads "$T" -v 2
stamp "START align"
"$MM" align "$X/fragdb" "$U" "$X/fragpref" "$X/fragaln" -e 1e-5 -a --threads "$T" -v 2
"$MM" convertalis "$X/fragdb" "$U" "$X/fragaln" "$X/hits.tsv" \
   --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
"$PY" -u "$C/tsv2parquet.py" "$X/hits.tsv" "$X/hits.parquet" keep_all
rm -f "$X/hits.tsv"
stamp "START score"
"$PY" -u "$C/cut_test_score.py" "$R" "$RD/subgenic"
stamp "START variants"
"$PY" -u "$C/build_variants.py" "$R" "$RD/subgenic" "$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet" "$RD/engine/dpann_tokens"
stamp "DONE"
