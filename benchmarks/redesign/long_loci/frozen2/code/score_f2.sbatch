#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf2_score
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/logs/score_%j.log
# long_loci frozen run 2: score every available output once (fscore2.py; protocol FROZEN2.json 'scoring').
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/code/env_f2.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID suffix=${SUFFIX:-}"
cd "$C"
step score${SUFFIX:-} "$PYE" -u fscore2.py score --threads "$T" --suffix "${SUFFIX:-}"
stamp "ALL DONE"
