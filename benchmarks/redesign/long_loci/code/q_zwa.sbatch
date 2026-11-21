#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ll_zwa
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_zwa_%j.log
# long_loci part 5: Z-WA on one dataset (DATA=dpann|sd): exclusion masks to distance 8 (gate: d <= 3 equals the stored
# masks), kernel gate at window 3 against vl.pass_count, adaptive round 0 in 4 orders, tau, L1 / L4b / L4b+.
# DPANN: then ODD scores of the 9 outputs.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID data=${DATA:?}"
cd "$C"
step zwa_$DATA "$PYT" -u zwa.py run --data "$DATA" --threads "$T"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$DATA" == dpann ]]; then
  step zwaodd "$PYE" -u dscore.py odd --tag dpann_ZWA --names L1,L4b_t0.02,L4b_t0.05,L4b_t0.1,L4b_t0.2,L4bp_t0.02,L4bp_t0.05,L4bp_t0.1,L4bp_t0.2 --threads "$T"
fi
stamp "ALL DONE"
