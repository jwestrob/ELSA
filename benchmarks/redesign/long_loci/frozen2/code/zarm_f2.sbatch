#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf2_zarm
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/logs/zarm_%j.log
# long_loci frozen run 2, J5b: PRIMARY Z arm (z >= 10.6 lists, vl round 0, list tau by vl's rule, L1 / L4b / L4b+ at
# theta .02 with kNN and token cohesion; the frozen output is L4bp_t0.02).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/code/env_f2.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
test -f "$FZ/logs/done_z_prep"
step z_arm "$PYE" -u fz2.py arm --arm Z --threads "$T"
stamp "ALL DONE"
