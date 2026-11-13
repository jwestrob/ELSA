#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J kinds_hom
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/kinds_hom_%j.log
# Coordinator-requested homology addition: family representatives searched against the uniqdb (MMseqs2 18, scales to
# all CPUs), then family homology tables, architecture groups, withheld passes with homologs (numba, all CPUs).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=$RD/ws_tokens/env/bin/python
U=$RD/subgenic/db/uniqdb
H=$RD/kinds/hom
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/kinds/tmp" NUMBA_CACHE_DIR="$RD/kinds/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$H"
cd "$RD/kinds/code"
stamp() { echo "$(date -Is) $*"; }
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
"$PY" -u homology.py reps --threads "$T"
if [[ ! -s "$H/hits.tsv" ]]; then
  rm -rf "$H/qdb"* "$H/aln"* "$H/mmtmp"
  "$MM" createsubdb "$H/reps_uid.txt" "$U" "$H/qdb" -v 2
  "$MM" createsubdb "$H/reps_uid.txt" "${U}_h" "$H/qdb_h" -v 2
  stamp "START search"
  "$MM" search "$H/qdb" "$U" "$H/aln" "$H/mmtmp" -s 7.5 -e 1e-3 --max-seqs 4000 --threads "$T" -v 2
  stamp "DONE search"
  "$MM" convertalis "$H/qdb" "$U" "$H/aln" "$H/hits.tsv" --format-output query,target,evalue --threads "$T" -v 2
  rm -rf "$H/mmtmp" "$H/aln".*
fi
wc -l "$H/hits.tsv"
for st in post arch withheld weval; do
  stamp "STEP $st"
  "$PY" -u homology.py "$st" --threads "$T"
done
stamp done
