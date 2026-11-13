#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ll_sdprep
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_sdprep_%j.log
# long_loci part 3: from the exact kNN (sd_knn): round-0 exclusion masks (group / genome lineage, 4 orders) and the kNN
# cohesion array kappa_var_knn50 (sdu.py prep).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
test -f "$LL/sd_knn/zca10__k2000__I.npy"
step sd_prep "$PYT" -u sdu.py prep --threads "$T"
stamp "ALL DONE"
