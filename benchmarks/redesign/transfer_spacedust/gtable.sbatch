#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -J ts_gtable
#SBATCH --exclude=node-224-2t-8gpu-1
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/logs/ts_gtable_%j.log
# transfer_spacedust: per-genome best matches (z table) and genome-pair embedding sums on stored zca10 vectors
# (A5000 node; linear algebra only, no language model). Then genome groups at 0.75.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/code/env.sh
GPY=/groups/banfield/scratch/projects/human/nano/int/kinsey/tools/Talea/.venv/bin/python
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
step gtable "$GPY" -u "$C/gtable.py"
step groups "$PY" -u "$C/emb.py" groups --threads "$T"
stamp "ALL DONE"
