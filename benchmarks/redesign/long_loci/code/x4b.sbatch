#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_x4b
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_x4b_%j.log
# long_loci L4b (amendment_2): builds on DPANN and Spacedust (groups), ODD scores of DPANN L4b only.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
step dp_rdiag_ref2 "$PYE" -u dscore.py rdiag --threads "$T"
step dp_build4b_group "$PYE" -u ll.py build4b --data dpann --count group --threads "$T"
step dp_odd4b_group "$PYE" -u dscore.py odd --tag dpann_group --names L4b_t0.02,L4b_t0.05,L4b_t0.1,L4b_t0.2 --out-suffix _L4b --threads "$T"
export PYTHONPATH="$SCALE/production:$SCALE/code"
step sd_build4b_group "$PYT" -u ll.py build4b --data sd --count group --threads "$T"
step sd_build4b_syn "$PYT" -u ll.py build4b --data sd --count syn --threads "$T"
stamp "ALL DONE"
