#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf_z
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/logs/z_%j.log
# long_loci frozen run, J5: SECONDARY. Adapter smoke, prep (kNN cohesion kappa_var_knn50, exclusion masks K200/K1000 for
# lineage group / genome, observed order + shuffles 0-2), then the Z arm (z >= 10.6 lists, vl round 0, tau by vl's rule,
# L1 / L4b / L4b+ at theta .02). fz.py = sdu.py with paths changed. 24 CPUs (coordinator, 2026-11-14); prep in zprep_f.sbatch.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/code/env_f.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
test -s "$FZ/knn/zca10__k2000__I.npy"; test -s "$FZ/emb/kappa_var_tokens.npy"
step z_smoke "$PYE" -u fz.py smoke
test -f "$FZ/logs/done_z_prep"   # prep runs in its own job (memory, see zprep_f.sbatch)
step z_arm "$PYE" -u fz.py arm --arm Z --threads "$T"
stamp "ALL DONE"
