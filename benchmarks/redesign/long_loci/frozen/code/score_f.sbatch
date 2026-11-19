#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf_score
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/logs/score_%j.log
# long_loci frozen run, J8: score every available output once (fscore.py; protocol FROZEN.json 'scoring'), then write the
# text tables (ftable.py) to results/tables_lut${SUFFIX}.txt. Run time is capped from outside by the login-node watchdog.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/code/env_f.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID suffix=${SUFFIX:-}"
cd "$C"
step score${SUFFIX:-} "$PYE" -u fscore.py score --threads "$T" --suffix "${SUFFIX:-}"
"$PYE" -u ftable.py "$FZ" "${SUFFIX:-}" > "$FZ/results/tables_lut${SUFFIX:-}.txt"
stamp "tables -> $FZ/results/tables_lut${SUFFIX:-}.txt"
stamp "ALL DONE"
