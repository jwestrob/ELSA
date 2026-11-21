#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -J eq_gpu
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/logs/%x_%j.log
# embed_queue worker on the general gpu partition (A5000 24 GB, or an H200 when one is free).
# The partition QOS allows one GPU job per user: 12 CPUs, 12 GB RAM.
# Usage: sbatch worker_a5000.sbatch <run dir> [asc|desc]
export EQ_SBATCH=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/worker_a5000.sbatch
exec bash /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/worker.sh "$@"
