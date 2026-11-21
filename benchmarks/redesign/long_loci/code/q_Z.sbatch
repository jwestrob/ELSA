#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ll_qZ
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/logs/ll_qZ_%j.log
# long_loci part 2: list-based round-0 bonds (vl.py kernels, read-only) for arms Z and Zs: ladders in the observed and 3
# shuffled orders, tau*, L1 / L4b / L4b+ with knn and token cohesion, family-free node filter. No scoring.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
step lb_Z "$PYT" -u lb.py arm --arm Z --fz --threads "$T"
step lb_Zs "$PYT" -u lb.py arm --arm Zs --threads "$T"
stamp "ALL DONE"
