#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_zwa2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_zwa2_%j.log
# long_loci part 5: DPANN EVEN of the selected Z-WA output (EV_NAMES), paired vs DH / part-1 L4b / part-2 Z / part-2 A /
# part-4 WA; then source207 of the primary (SD_PRIMARY) and DEVELOPMENT rows (SD_DEV).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID ev=${EV_NAMES:?} sd=${SD_PRIMARY:?} dev=${SD_DEV:-}"
cd "$C"
CMP="p1L4b=dpann_group/L4b_t0.1,p2Z=dpann_Z/L4bp_t0.02,p2A=dpann_A/L4b_t0.02,p4WA=dpann_WA/L4b_t0.2"
step zwa_even "$PYE" -u dscore.py even --tag dpann_ZWA --names "$EV_NAMES" --compare "$CMP" --extra --out-suffix _p5 --threads "$T"
step zwa_pairs "$PYE" -u dscore.py pairs --tag p5 --pairs "${PAIRS:?}" --threads "$T"
export PYTHONPATH="$SCALE/production:$SCALE/code"
REF="p1L4b=sd_group/pr_L4b_t0.1,p1L1=sd_group/pr_L1,p3Z=sd_Z/pr_L4bp_t0.02,p4WA=sd_WA/pr_L4b_t0.2"
step zwa_s207 "$PYT" -u sscore.py --tag sd_ZWA --names "$SD_PRIMARY" --extra-refs "$REF" --label PRIMARY --threads "$T"
if [[ -n "${SD_DEV:-}" ]]; then step zwa_s207dev "$PYT" -u sscore.py --tag sd_ZWA --names "$SD_DEV" --extra-refs "$REF" --label DEVELOPMENT --out-suffix _dev --threads "$T"; fi
stamp "ALL DONE"
