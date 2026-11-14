#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J c1x_coup2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/candidate1/logs/coup2_%j.log
# Task 4 step 3: within-order and within-genus pairing-shuffle nulls; top-ranked inter-protein pairs.
set -euo pipefail
X=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/candidate1
echo "threads $SLURM_CPUS_ON_NODE host $(hostname) $(date)"
cd $X/code && /home/jwestrob/miniconda3/envs/elsa/bin/python $X/code/c1x_coupling2.py $X/coup $SLURM_CPUS_ON_NODE
echo "done $(date)"
