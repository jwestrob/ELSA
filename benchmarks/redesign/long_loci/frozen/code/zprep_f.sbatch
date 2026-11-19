#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 40
#SBATCH -J llf_zprep
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/logs/zprep_%j.log
# long_loci frozen run, J5a: SECONDARY prep only, 40 CPUs / 160 GB (Spacedust prep peaked at 120 GB; lut ~126-135 GB),
# CPU count chosen for memory (approved by the coordinator 2026-11-14). fzrun.py prep: fz.py unchanged (see fzrun.py). SECONDARY. Adapter smoke, prep (kNN cohesion kappa_var_knn50, exclusion masks K200/K1000 for
# lineage group / genome, observed order + shuffles 0-2), then the Z arm (z >= 10.6 lists, vl round 0, tau by vl's rule,
# L1 / L4b / L4b+ at theta .02). fz.py = sdu.py with paths changed. 24 CPUs (coordinator, 2026-11-14); prep in zprep_f.sbatch.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/code/env_f.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
cd "$C"
test -s "$FZ/knn/zca10__k2000__I.npy"; test -f "$FZ/logs/done_f_zca32"; test -f "$FZ/logs/done_f_groups"
step z_prep "$PYE" -u fzrun.py prep --threads "$T"
stamp "ALL DONE"
