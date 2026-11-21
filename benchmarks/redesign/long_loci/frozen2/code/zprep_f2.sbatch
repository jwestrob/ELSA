#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 40
#SBATCH -J llf2_zprep
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/logs/zprep_%j.log
# long_loci frozen run 2, J5a: PRIMARY prep (adapter smoke, kNN cohesion, exclusion masks). 40 CPUs FOR MEMORY (lut prep
# peaked at 126 GB for 4.46 M uids; 3.17 M uids here, ~90-100 GB expected, above the 96 GB of 24 CPUs).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/code/env_f2.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
test -s "$FZ/knn/zca10__k2000__I.npy"; test -s "$FZ/emb/kappa_var_tokens.npy"
step z_smoke "$PYE" -u fz2.py smoke
step z_prep "$PYE" -u fz2.py prep --threads "$T"
stamp "ALL DONE"
