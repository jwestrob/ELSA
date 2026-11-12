#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J c1x_coup
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/candidate1/logs/coup_%j.log
# Task 4: B-C and A-B coupling (MIp and mean-field DCA) on the deep-dive instance alignments, contact vs non-contact
# column pairs from the ABC models, pairing-shuffle nulls (global and within phylum); one process per CPU.
set -euo pipefail
X=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/candidate1
echo "threads $SLURM_CPUS_ON_NODE host $(hostname) $(date)"
/home/jwestrob/miniconda3/envs/elsa/bin/python $X/code/c1x_coupling.py $X/coup $SLURM_CPUS_ON_NODE
echo "done $(date)"
