#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu_h200
#SBATCH --qos=gpu_h200_4
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH -J eq_h200
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/logs/%x_%j.log
# embed_queue worker on one H200. Usage: sbatch worker_h200.sbatch <run dir> [asc|desc]
# Keep it at ONE submitted job under gpu_h200_4 while the Talea lanes run (10 submitted jobs per user; check with the
# Talea session first). This worker never submits another job under that QOS.
export EQ_SBATCH=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/worker_h200.sbatch EQ_RESUBMIT=0 EQ_ON_BAD_GPU=work
exec bash /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/worker.sh "$@"
