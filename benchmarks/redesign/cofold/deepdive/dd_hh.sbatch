#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dd_hh
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive/logs/hh_%x_%j.log
# Candidate-1 profile searches (code/dd_hh.py). Usage: sbatch -J dd_hh_u --export=ALL,MODE=u dd_hh.sbatch (MODE u|ub)
set -euo pipefail
export PATH=/shared/software/bin:$PATH
DD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive
cd $DD
echo "mode $MODE threads $SLURM_CPUS_ON_NODE host $(hostname) $(date)"
/home/jwestrob/miniconda3/envs/elsa/bin/python code/dd_hh.py $MODE
echo "done $(date)"
