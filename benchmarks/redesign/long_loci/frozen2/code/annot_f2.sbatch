#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf2_annot
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/logs/annot_%j.log
# long_loci frozen run 2: restricted annotation (KOfam ribosomal-protein + N_RULES KOs; N_RULES / EF2 Pfam at GA) for the
# audit's accurate reading (annot2.py). One hmmsearch process per CPU.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/code/env_f2.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
step annot "$PYE" -u annot2.py
stamp "ALL DONE"
