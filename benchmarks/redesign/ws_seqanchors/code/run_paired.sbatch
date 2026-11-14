#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sa_paired
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors/logs/paired_%j.log
# Paired region bootstrap of source207 F1 excess differences between arms.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/home/jwestrob/miniconda3/envs/elsa/bin/python -u /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_seqanchors/code/compare.py paired --threads "${SLURM_CPUS_ON_NODE:?}"
