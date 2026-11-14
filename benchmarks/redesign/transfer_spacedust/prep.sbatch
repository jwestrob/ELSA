#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ts_prep
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/logs/ts_prep_%j.log
# transfer_spacedust prep: learner port check on DPANN, zca10 vectors, reference reproduction, engine rerun.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
step refs "$PY" -u "$C/refs.py" rescore --threads "$T"
step portcheck "$PY" -u "$C/portcheck.py"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
step zca "$PY" -u "$C/emb.py" zca --threads "$T"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
md5sum "$C/recurrence.py"
step engine_rerun /home/jwestrob/miniconda3/envs/elsa/bin/python -u "$C/recurrence.py" \
    --arm hybrid__esm2_650m__genomecentred__mm30cpmr0p05 --w 8 --null genome --out "$TS/engine" --threads "$T"
step engine_compare "$PY" -u "$C/refs.py" compare --rdir "$TS/engine/hybrid__esm2_650m__genomecentred__mm30cpmr0p05__w8"
stamp "ALL DONE"
