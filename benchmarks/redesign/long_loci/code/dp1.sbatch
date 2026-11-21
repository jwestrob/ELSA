#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_dp1
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_dp1_%j.log
# long_loci DPANN stage 1: reference reproduction, bonds / L1-L4 build (groups), ODD scoring only, reference ribosomal
# diagnosis (token DH and stored word outputs), then the syn-counting build and its odd scores (secondary).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
step dp_refcheck "$PYE" -u dscore.py refcheck --threads "$T"
step dp_build_group "$PYE" -u ll.py build --data dpann --count group --threads "$T"
step dp_odd_group "$PYE" -u dscore.py odd --tag dpann_group --threads "$T"
step dp_rdiag_ref "$PYE" -u dscore.py rdiag --threads "$T"
step dp_build_syn "$PYE" -u ll.py build --data dpann --count syn --threads "$T"
step dp_odd_syn "$PYE" -u dscore.py odd --tag dpann_syn --threads "$T"
stamp "ALL DONE"
