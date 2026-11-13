#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_qodd
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_qodd_%j.log
# long_loci part 2: ODD scores of L1 / L4b / L4b+ (knn cohesion) for every list arm, F's L4b+, and the selection.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
N=L1,L4b_t0.02,L4b_t0.05,L4b_t0.1,L4b_t0.2,L4bp_t0.02,L4bp_t0.05,L4bp_t0.1,L4bp_t0.2
if [[ "${DO_F:-0}" == 1 ]]; then step odd_F_L4bp "$PYE" -u dscore.py odd --tag dpann_group --names L4bp_t0.02,L4bp_t0.05,L4bp_t0.1,L4bp_t0.2 --out-suffix _L4bp --threads "$T"; fi
for A in ${ODD_ARMS:?}; do
  step odd_$A "$PYE" -u dscore.py odd --tag dpann_$A --names "$N" --threads "$T"
done
"$PYE" sel2.py "$LL/results"
stamp "ALL DONE"
