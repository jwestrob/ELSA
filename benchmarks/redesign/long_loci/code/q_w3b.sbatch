#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_w3b
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_w3b_%j.log
# long_loci part 4 amendment_1: W3b (window 3 at its bond-rule tau) on DPANN (+ odd scores) and Spacedust.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
step w3b_dpann "$PYE" -u lw.py build --data dpann --arms W3 --w3b --suffix _W3b --threads "$T"
step wodd_W3b "$PYE" -u dscore.py odd --tag dpann_W3b --names L1,L4b_t0.02,L4b_t0.05,L4b_t0.1,L4b_t0.2 --threads "$T"
step w3b_sd "$PYE" -u lw.py build --data sd --arms W3 --w3b --suffix _W3b --threads "$T"
stamp "ALL DONE"
