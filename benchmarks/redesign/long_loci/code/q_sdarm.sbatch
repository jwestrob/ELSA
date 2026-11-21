#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ll_sdarm
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_sdarm_%j.log
# long_loci part 3: list-bond arms on the Spacedust collection (sdu.py arm), then source207 scores of the predeclared
# outputs with the token-cohesion checks. SD_ARMS = e.g. "A As" (needs mm_sd arrays + sd_prep) or "Z Zs" (needs the z table).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
export PYTHONPATH="$SCALE/production:$SCALE/code"
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID arms=${SD_ARMS:?}"
cd "$C"
test -f "$LL/logs/done_sd_prep"
REF="p1L4b=sd_group/pr_L4b_t0.1,p1L4b02=sd_group/pr_L4b_t0.02,p1L1=sd_group/pr_L1"
for A in $SD_ARMS; do
  step sdarm_$A "$PYT" -u sdu.py arm --arm $A --threads "$T"
  case $A in A|As) N=L4b_t0.02,L4btok_t0.02,L4bp_t0.02,L1;; Z|Zs) N=L4bp_t0.02,L4bptok_t0.02,L4b_t0.02,L1;; esac
  step sds207_$A "$PYT" -u sscore.py --tag sd_$A --names "$N" --extra-refs "$REF" --threads "$T"
done
stamp "ALL DONE"
