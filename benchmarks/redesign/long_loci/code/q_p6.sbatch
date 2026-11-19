#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_p6
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/p6_%j.log
# long_loci part 6 (predeclared6.json). MODE=dpann: bridges + gates + variant builds for both bases (DPANN);
# MODE=scoreP / scoreZ: DPANN scoring, odd selection, even report; MODE=sdbr: Spacedust bridges; MODE=sd: Spacedust
# builds + source207 (DEVELOPMENT). Run time capped from outside by the login-node watchdog.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID mode=${MODE:?}"
cd "$C"
case "$MODE" in
  dpann)  step p6_br_dpann "$PYE" -u p6.py bridges --data dpann --threads "$T"
          step p6_build_dpann_P "$PYE" -u p6.py build --data dpann --base P --threads "$T"
          step p6_build_dpann_Z "$PYE" -u p6.py build --data dpann --base Z --threads "$T" ;;
  scoreP) step p6_score_P "$PYE" -u p6.py score --base P --threads "$T" ;;
  scoreZ) step p6_score_Z "$PYE" -u p6.py score --base Z --threads "$T" ;;
  sdbr)   step p6_br_sd "$PYE" -u p6.py bridges --data sd --threads "$T" ;;
  sd)     step p6_sd "$PYE" -u p6.py sd --threads "$T" ;;
esac
stamp "ALL DONE"
