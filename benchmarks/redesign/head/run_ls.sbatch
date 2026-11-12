#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J head_ls
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/head_ls_%j.log
# Locus search with the chosen student's ranks. usage: STAGES=topk,selfcheck,tune,final,evidence,summary sbatch --export=ALL run_ls.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue
export TMPDIR=$RD/head/tmp NUMBA_CACHE_DIR=$RD/head/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$RD/head/ls"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID stages=${STAGES:-topk,selfcheck,tune,final,evidence,summary}"
"$PY" -u "$RD/head/code/head_ls.py" "${STAGES:-topk,selfcheck,tune,final,evidence,summary}" --threads "$T"
