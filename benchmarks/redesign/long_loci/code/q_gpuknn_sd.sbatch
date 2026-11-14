#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -J ll_gknnsd
#SBATCH --exclude=node-224-2t-8gpu-1
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_gknnsd_%j.log
# long_loci part 3: exact cosine kNN (k 2,000) of the Spacedust unique zca10 vectors on one A5000 (gpu_knn_sd.py);
# KARGS="--limit 32768" times the first queries only. A5000 partition only (H200 node excluded).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=/groups/banfield/scratch/projects/human/nano/int/kinsey/tools/Talea/.venv/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-none} args=${KARGS:-}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -u "$RD/long_loci/code/gpu_knn_sd.py" --k 2000 ${KARGS:-}
echo "$(date -Is) done"
