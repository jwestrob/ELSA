#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_p7
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/p7_%j.log
# long_loci part 7 (predeclared7.json). MODE=build: mechanism check on part-6 S outputs, then S-add / S-add+B builds for
# both bases (DPANN); MODE=scoreP / scoreZ: DPANN scoring; MODE=sd: Spacedust rows (DEVELOPMENT). Watchdog caps outside.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID mode=${MODE:?}"
cd "$C"
case "$MODE" in
  build)  step p7_mech "$PYE" -u p7.py mech --threads "$T"
          step p7_build_dpann_Z "$PYE" -u p7.py build --data dpann --base Z --threads "$T"
          step p7_build_dpann_P "$PYE" -u p7.py build --data dpann --base P --threads "$T" ;;
  scoreP) step p7_score_P "$PYE" -u p7.py score --base P --threads "$T" ;;
  scoreZ) step p7_score_Z "$PYE" -u p7.py score --base Z --threads "$T" ;;
  sd)     step p7_sd "$PYE" -u p7.py sd --threads "$T" ;;
esac
stamp "ALL DONE"
