#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J rp_arms
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/repair_arms_%j.log
# Workstream repair: split stage 1 (candidates from the shorter piece's hits), stage 2 (longer piece aligned to each
# candidate X through an explicit prefilter database), then both repair rules and the token arms.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
R=$RD/repair; C=$R/code; S=$R/split; U=$RD/subgenic/db/uniqdb
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T"
if [[ -z "${SKIP_STAGE2:-}" ]]; then
  "$PY" -u "$C/split_candidates.py" "$R" "$RD/subgenic"
  wc -l "$S/pref.tsv"
  "$MM" rmdb "$S/pref" -v 1 2>/dev/null || true; "$MM" rmdb "$S/aln" -v 1 2>/dev/null || true
  "$MM" tsv2db "$S/pref.tsv" "$S/pref" --output-dbtype 7 -v 2
  stamp "START stage-2 align"
  "$MM" align "$U" "$U" "$S/pref" "$S/aln" -e 1e-5 -a --alignment-mode 3 --threads "$T" -v 2
  stamp "DONE stage-2 align"
  "$MM" convertalis "$U" "$U" "$S/aln" "$S/stage2.tsv" \
     --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
  "$PY" -u "$C/tsv2parquet.py" "$S/stage2.tsv" "$S/stage2.parquet"
  rm -f "$S/stage2.tsv"
fi
stamp "START arms"
"$PY" -u "$C/build_arms.py" "$R" "$RD/subgenic" "$RD/ws_dpann/package" \
    "$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet" "$RD/engine/dpann_tokens"
stamp "DONE arms"
