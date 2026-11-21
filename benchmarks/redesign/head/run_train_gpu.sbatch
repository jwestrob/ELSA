#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -J head_train
#SBATCH --exclude=node-224-2t-8gpu-1,node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/head_train_%j.log
# Student head training on one A5000 (12 CPUs, 12 GB host memory: vectors are memory-mapped and moved in chunks).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/groups/banfield/scratch/projects/human/nano/int/kinsey/tools/Talea/.venv/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -u "$RD/head/code/head_train.py" --data "$RD/head/data" --out "$RD/head/student" --device cuda \
    --batch 4096 --steps 12000 --eval-every 1500
