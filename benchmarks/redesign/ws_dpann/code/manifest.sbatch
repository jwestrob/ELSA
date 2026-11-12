#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dpann_manifest
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/manifest_%j.log
# Hash and count every package file (parallel over files).
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
cp $W/code/*.py $W/code/*.sbatch $W/package/code/
rm -f $W/package/code/score_dpann_loci.py $W/package/code/selftest_scorer.py
/home/jwestrob/miniconda3/envs/elsa/bin/python $W/code/make_manifest.py --pkg $W/package --threads "$SLURM_CPUS_ON_NODE"
