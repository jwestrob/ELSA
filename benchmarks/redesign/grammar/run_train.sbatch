#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J gr_train
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/grammar_train_%j.log
# usage: SPLIT=cv0 [EXTRA="--limit-genomes 300 --max-epochs 1"] sbatch --export=ALL run_train.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T TORCH_NUM_THREADS=$T
export TMPDIR=$RD/tmp
cd $RD/grammar/code
echo "split=${SPLIT:?} extra=${EXTRA:-} threads=$T host=$(hostname)"
"$PY" -u train.py --split "$SPLIT" --threads "$T" ${EXTRA:-}
