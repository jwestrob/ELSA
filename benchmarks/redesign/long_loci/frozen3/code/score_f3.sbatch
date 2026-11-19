#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf3_score
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen3/logs/score_%j.log
# long_loci frozen run 3: ribosomal standing counts + descriptive statistics + locus-table manifest (fscore3.py).
# RUNS = the ribosomal runs table (package truth or runs built from the audit agent's annotation).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen3/code/env_f3.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID runs=${RUNS:?}"
cd "$C"
step score "$PYE" -u fscore3.py score --threads "$T" --runs "$RUNS"
stamp "ALL DONE"
