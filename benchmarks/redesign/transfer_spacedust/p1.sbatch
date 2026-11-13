#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ts_p1
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/logs/ts_p1_%j.log
# transfer_spacedust P1: group-unit word ladder (+ genome-unit ladder, diagnostic, for the stress test's tau), readings and
# source207 scores; then the secondary duplicate-genome stress test (amendment_0).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -f "$TS/logs/done_groups" && test -f "$TS/logs/done_cohesion"
step ladder_group "$PY" -u "$C/words.py" ladder --specs group,genome --threads "$T" --tag group_genome
TAU=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_group_genome.json'))['results']['group']['tau_star'])")
TAUG=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_group_genome.json'))['results']['genome']['tau_star'])")
TAUS=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_syn.json'))['results']['syn']['tau_star'])")
stamp "P1 tau* = $TAU; genome tau* = $TAUG; P2 tau* = $TAUS"
step score_P1 "$PY" -u "$C/wscore.py" --spec group --tau "$TAU" --tag P1 --compare "P2:V-coh0.1,P2:CC1,P2:CC2" --threads "$T"
if [[ "${STRESS:-1}" == 1 ]]; then
  step stress "$PY" -u "$C/stress.py" --taus "genome:$TAUG,group:$TAU,syn:$TAUS" --threads "$T"
fi
stamp "ALL DONE"
