#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_dp2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_dp2_%j.log
# long_loci DPANN stage 2 (after the odd selection is recorded): EVEN verdicts of the selected outputs, paired intervals,
# ribosomal diagnosis of each output. NAMES_GROUP / NAMES_SYN passed with --export.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID names_group=${NAMES_GROUP:?} names_syn=${NAMES_SYN:-}"
cd "$C"
step dp_build4b_syn "$PYE" -u ll.py build4b --data dpann --count syn --threads "$T"
step dp_mkplus "$PYE" -u mkplus.py dpann_group dpann_syn
step dp_even_group "$PYE" -u dscore.py even --tag dpann_group --names "$NAMES_GROUP" --rdiag --threads "$T"
if [[ -n "${NAMES_SYN:-}" ]]; then step dp_even_syn "$PYE" -u dscore.py even --tag dpann_syn --names "$NAMES_SYN" --threads "$T"; fi
stamp "ALL DONE"
