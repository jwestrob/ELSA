#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_qeven
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_qeven_%j.log
# long_loci part 2: EVEN verdicts of the odd-selected output per arm (+ token-cohesion check), paired vs DH, part-1 L4b,
# Z vs A; CRISPR-Cas members outside loci; the 56 runs. EV_<arm> = comma-separated names; PAIRS for dscore pairs.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
for A in F Z A Zs As FZ; do
  v="EV_$A"; names="${!v:-}"; [[ -z "$names" ]] && continue
  tag=dpann_$A; [[ "$A" == F ]] && tag=dpann_group
  step even2_$A "$PYE" -u dscore.py even --tag "$tag" --names "$names" --compare p1L4b=dpann_group/L4b_t0.1 --extra --out-suffix _p2 --threads "$T"
done
step pairs2 "$PYE" -u dscore.py pairs --tag p2 --pairs "${PAIRS:?}" --threads "$T"
stamp "ALL DONE"
