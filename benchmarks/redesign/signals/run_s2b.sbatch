#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sig_s2e30
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/signals/logs/sig_%j.log
# usage: sbatch --export=ALL,SIG_EPOCHS=30,SIG_TAG=_e30 run_s2.sbatch   (signal 2: classifier, tests 2-4, operon)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export SIG_THREADS=$T PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export NUMBA_NUM_THREADS=$T TMPDIR=$RD/tmp
cd $RD/signals/code
echo "job $SLURM_JOB_ID host $(hostname) cpus $T script s2 start $(date -u +%FT%TZ)"
"$PY" -u s2_classify.py
"$PY" -u s2_analyze.py
"$PY" -u s2_operon.py
# the predeclared 4-epoch model again with the post hoc additions (strand splits, genome-centred foreignness)
cp -n $RD/signals/work/s2_tests.json $RD/signals/work/s2_tests_firstpass.json
SIG_TAG= "$PY" -u s2_analyze.py
SIG_TAG= "$PY" -u s2_operon.py
echo "end $(date -u +%FT%TZ)"
