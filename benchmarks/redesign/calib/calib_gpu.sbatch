#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -J calib_gpu
#SBATCH --exclude=node-224-2t-8gpu-1
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/calib_gpu_%j.log
# ws_calib GPU linear algebra on stored zca10 vectors (A5000 node; no language-model inference).
# usage: sbatch --export=ALL,GSCRIPT=gpu_feat.py calib_gpu.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/groups/banfield/scratch/projects/human/nano/int/kinsey/tools/Talea/.venv/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -u "$RD/calib/code/${GSCRIPT:?}"
echo "$(date -Is) done"
