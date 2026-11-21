#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -J llf2_gpu
#SBATCH --exclude=node-224-2t-8gpu-1
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/logs/gpu_%j.log
# long_loci frozen run 2, J2 / J3 (A5000 partition only, never gpu_h200; linear algebra on the package's stored zca10
# vectors, no language-model inference):
#   MODE=table  gtable_f2.py (transfer_spacedust gtable.py copied): z table (Z_STORE 10.5) + genome-pair sums
#   MODE=knn    gpu_knn_f2.py (gpu_knn_sd.py copied): exact cosine kNN, k 2,000
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/code/env_f2.sh
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T Z_STORE=10.5
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID gpus=${CUDA_VISIBLE_DEVICES:-none} mode=${MODE:?}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
case "$MODE" in
  table) step gpu_table "$GPY" -u "$C/gtable_f2.py" ;;
  knn)   step gpu_knn "$GPY" -u "$C/gpu_knn_f2.py" --k 2000 ;;
esac
stamp "ALL DONE"
