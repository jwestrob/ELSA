#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J head_train_cpu
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/head_train_cpu_%j.log
# CPU fallback of the student head training (GPU slot not started within 60 minutes).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/groups/banfield/scratch/projects/human/nano/int/kinsey/tools/Talea/.venv/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
"$PY" -u "$RD/head/code/head_train.py" --data "$RD/head/data" --out "$RD/head/student" --device cpu \
    --batch 2048 --steps 6000 --eval-every 1000
