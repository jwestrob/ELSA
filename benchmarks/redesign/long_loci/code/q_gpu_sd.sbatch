#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -J ll_gpusd
#SBATCH --exclude=node-224-2t-8gpu-1
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_gpusd_%j.log
# long_loci part 3: ws_calib z table for the Spacedust collection (calib_gpu.sbatch copied; A5000 partition, never
# gpu_h200; linear algebra on stored zca10 vectors, no language-model inference). Z_STORE 10.5 as the DPANN table.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/groups/banfield/scratch/projects/human/nano/int/kinsey/tools/Talea/.venv/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T Z_STORE=10.5
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -u "$RD/long_loci/code/gpu_table_sd.py"
echo "$(date -Is) done"
