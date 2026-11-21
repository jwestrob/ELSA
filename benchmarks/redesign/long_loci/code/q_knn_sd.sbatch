#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ll_knnsd
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_knnsd_%j.log
# long_loci part 3: HNSW cosine kNN (k 2,000) of the Spacedust unique zca10 vectors (softrec knn_deep.py recipe, CPU),
# then the round-0 exclusion masks (group / genome lineage, 4 orders) and the kNN cohesion array kappa_var_knn50.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
step sd_knn "$PYT" -u knn_sd.py --k 2000 --threads "$T"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
step sd_prep "$PYT" -u sdu.py prep --threads "$T"
stamp "ALL DONE"
