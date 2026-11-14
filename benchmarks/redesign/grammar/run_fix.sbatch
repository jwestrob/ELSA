#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J gr_fix
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/grammar_fix_%j.log
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue TMPDIR=$RD/tmp NUMBA_CACHE_DIR=$RD/tmp/numba_gr_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
cd $RD/grammar/code
$RD/ws_tokens/env/bin/python -u fix_nlogp.py "$T"
