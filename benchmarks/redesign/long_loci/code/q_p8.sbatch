#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_p8
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/p8_%j.log
# long_loci part 8 (predeclared8.json): best-window bonds. MODE = pbw_dpann | zbw_dpann | pbw_sd | zbw_sd | scoreP |
# scoreZ | sd. Run time capped from outside by the login-node watchdog.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID mode=${MODE:?}"
cd "$C"
case "$MODE" in
  pbw_dpann) step p8_pbw_dpann "$PYE" -u p8.py pbw --data dpann --threads "$T" ;;
  zbw_dpann) step p8_zbw_dpann "$PYE" -u p8.py zbw --data dpann --threads "$T" ;;
  pbw_sd)    step p8_pbw_sd "$PYE" -u p8.py pbw --data sd --threads "$T" ;;
  zbw_sd)    step p8_zbw_sd "$PYE" -u p8.py zbw --data sd --threads "$T" ;;
  scoreP)    step p8_score_P "$PYE" -u p8.py score --base P --threads "$T" ;;
  scoreZ)    step p8_score_Z "$PYE" -u p8.py score --base Z --threads "$T" ;;
  sd)        step p8_sd "$PYE" -u p8.py sd --threads "$T" ;;
esac
stamp "ALL DONE"
