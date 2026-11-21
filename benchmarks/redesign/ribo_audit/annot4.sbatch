#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ribo4_annot
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit/work4/annot4_%j.log

# ribo_audit part 4: part-3 restricted annotation (226 KOs + 20 Pfam over all proteins; 16 bacterial-member KOs over all
# proteins; Pfam-A on run windows) for the lut test set and DPANN (annot4.py). One pool of $SLURM_CPUS_ON_NODE
# single-CPU hmmsearch processes over round-robin protein chunks.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
cd /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit
echo "CPUs: $SLURM_CPUS_ON_NODE"
/usr/bin/time -v /home/jwestrob/miniconda3/envs/elsa/bin/python -u code/annot4.py
