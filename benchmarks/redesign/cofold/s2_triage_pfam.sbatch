#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J cofold2_pfam
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s2/logs/pfam_%j.log
# Remote-homology triage of the screen-2 and Part A block families (see code/s2_triage_remote.py).
set -euo pipefail
export PATH=/shared/software/bin:$PATH
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
S2=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s2
cd $S2
echo "threads $SLURM_CPUS_ON_NODE host $(hostname) $(date)"
$PY code/s2_triage_remote.py pfam
$PY code/s2_triage_remote.py selftest
echo "done $(date)"
