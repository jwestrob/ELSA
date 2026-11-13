#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_w
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_w_%j.log
# long_loci part 4: family bonds at pair distances W3 (ladder check) / W5 / W8 / WA on one dataset (DATA=dpann|sd):
# bond-FDR tau ladders on the observed order and 3 token shuffles, L1 / L4b / L4b+ builds. For DPANN, then ODD scores.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID data=${DATA:?}"
cd "$C"
step wbuild_$DATA "$PYE" -u lw.py build --data "$DATA" --threads "$T"
if [[ "$DATA" == dpann ]]; then
  for A in W5 W8 WA; do
    step wodd_$A "$PYE" -u dscore.py odd --tag dpann_$A --names L1,L4b_t0.02,L4b_t0.05,L4b_t0.1,L4b_t0.2 --threads "$T"
  done
fi
stamp "ALL DONE"
