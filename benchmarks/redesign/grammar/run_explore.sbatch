#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J gr_explore
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/grammar_explore_%j.log
# exploratory, after the final evaluation: cosine-held-fixed evaluation 1, and evaluation 5 without paralog-like pairs.
# The two parts are independent and run side by side; each uses its share of the allocated CPUs.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
H=$(( T / 2 ))
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$RD/tmp NUMBA_CACHE_DIR=$RD/tmp/numba_gr_${SLURM_JOB_ID} NUMBA_THREADING_LAYER=workqueue
mkdir -p "$NUMBA_CACHE_DIR"
cd $RD/grammar/code
OMP_NUM_THREADS=$H "$PY" -u explore_cos.py > $RD/grammar/eval/explore_cos.log 2>&1 &
p1=$!
OMP_NUM_THREADS=$(( T - H )) MKL_NUM_THREADS=$(( T - H )) NUMBA_NUM_THREADS=$(( T - H )) GR_LEARN_MAXCOS=0.839 GR_LEARN_SUFFIX=_lowcos \
    "$PY" -u evaluate.py learned "$(( T - H ))" > $RD/grammar/eval/learned_lowcos.log 2>&1 &
p2=$!
wait $p1; wait $p2
echo done
