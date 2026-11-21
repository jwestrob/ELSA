#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ll_sd2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_sd2_%j.log
# long_loci Spacedust scoring (DPANN-selected settings; other thetas as labelled sensitivity). Names via --export.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
export PYTHONPATH="$SCALE/production:$SCALE/code"
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
step sd_mkplus "$PYT" -u mkplus.py sd_group sd_syn
step sd_s207_group "$PYT" -u sscore.py --tag sd_group --names "${SD_GROUP:?}" --threads "$T"
if [[ -n "${SD_SYN:-}" ]]; then step sd_s207_syn "$PYT" -u sscore.py --tag sd_syn --names "$SD_SYN" --threads "$T"; fi
if [[ -n "${SD_SENS:-}" ]]; then step sd_s207_sens "$PYT" -u sscore.py --tag sd_group --names "$SD_SENS" --label SENSITIVITY --out-suffix _sens --threads "$T"; fi
stamp "ALL DONE"
