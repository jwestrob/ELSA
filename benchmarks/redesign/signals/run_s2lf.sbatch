#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sig_s2lf
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/signals/logs/sig_%j.log
# usage: sbatch run_s2lf.sbatch   (post hoc label-free signal 2: background-subspace projection, then tests 2-4)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export SIG_THREADS=$T PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export NUMBA_NUM_THREADS=$T TMPDIR=$RD/tmp
cd $RD/signals/code
echo "job $SLURM_JOB_ID host $(hostname) cpus $T script s2lf start $(date -u +%FT%TZ)"
"$PY" -u s2_labelfree.py
SIG_TAG=_lf "$PY" -u s2_analyze.py
echo "end $(date -u +%FT%TZ)"
