#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ribo3_annot
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit/work3/annot3_%j.log

# ribo_audit part 3: bacterial-member KOfam search over all SR-VP proteins + Pfam-A on the run windows (annot3.py).
# One hmmsearch process per allocated CPU ($SLURM_CPUS_ON_NODE chunks).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
cd /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit
/usr/bin/time -v /home/jwestrob/miniconda3/envs/elsa/bin/python -u code/annot3.py
