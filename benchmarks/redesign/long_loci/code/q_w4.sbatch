#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_w4
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_w4_%j.log
# long_loci part 4: DPANN EVEN (selected WA L4b .2 + labelled rows, paired vs DH and part-1 L4b), then source207 of the
# primary (WA L4b .2) and DEVELOPMENT rows.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
CMP="p1L4b=dpann_group/L4b_t0.1"
step w4even_WA "$PYE" -u dscore.py even --tag dpann_WA --names L4b_t0.2,L4b_t0.1 --compare "$CMP" --extra --out-suffix _p4 --threads "$T"
step w4even_W3b "$PYE" -u dscore.py even --tag dpann_W3b --names L4b_t0.02 --compare "$CMP" --extra --out-suffix _p4 --threads "$T"
step w4even_W5 "$PYE" -u dscore.py even --tag dpann_W5 --names L4b_t0.02 --compare "$CMP" --extra --out-suffix _p4 --threads "$T"
step w4even_W8 "$PYE" -u dscore.py even --tag dpann_W8 --names L4b_t0.1 --compare "$CMP" --extra --out-suffix _p4 --threads "$T"
step w4pairs "$PYE" -u dscore.py pairs --tag p4 --pairs "dpann_WA/L4b_t0.2:dpann_group/L4b_t0.1,dpann_WA/L4b_t0.1:dpann_group/L4b_t0.1,dpann_W3b/L4b_t0.02:dpann_group/L4b_t0.1,dpann_W5/L4b_t0.02:dpann_group/L4b_t0.1,dpann_W8/L4b_t0.1:dpann_group/L4b_t0.1,dpann_WA/L4b_t0.2:dpann_W3b/L4b_t0.02" --threads "$T"
export PYTHONPATH="$SCALE/production:$SCALE/code"
REF="p1L4b=sd_group/pr_L4b_t0.1,p1L1=sd_group/pr_L1"
step w4s207_WA "$PYT" -u sscore.py --tag sd_WA --names L4b_t0.2,L1 --extra-refs "$REF" --threads "$T"
for A in W3b W5 W8; do
  step w4s207_$A "$PYT" -u sscore.py --tag sd_$A --names L4b_t0.2,L1 --extra-refs "$REF" --label DEVELOPMENT --threads "$T"
done
step w4s207_W3 "$PYT" -u sscore.py --tag sd_group --names L4b_t0.2 --extra-refs "$REF" --label REFERENCE --out-suffix _p4 --threads "$T"
stamp "ALL DONE"
