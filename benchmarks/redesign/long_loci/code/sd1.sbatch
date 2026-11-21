#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_sd1
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_sd1_%j.log
# long_loci Spacedust stage 1 (no scoring): group-unit word ladder (copied learner, tau* by the 1% word-type FDR rule)
# -> bonds / L1-L4 build under groups at tau*; build under syn at P2's tau 8 (L1 checked equal to stored CC2).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
export PYTHONPATH="$SCALE/production:$SCALE/code"
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
step sd_ladder_group "$PYT" -u words_ts.py ladder --specs group --taus 4,6,8,10,12,15,20 --nulls 3 --threads "$T" --tag group
TAU=$("$PYT" -c "import json; print(json.load(open('$LL/results/ladder_group.json'))['results']['group']['tau_star'])")
stamp "group tau* = $TAU"
step sd_build_group "$PYT" -u ll.py build --data sd --count group --tau "$TAU" --threads "$T"
step sd_build_syn "$PYT" -u ll.py build --data sd --count syn --threads "$T"
stamp "ALL DONE"
