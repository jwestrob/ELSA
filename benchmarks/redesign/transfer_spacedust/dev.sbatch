#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ts_dev
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/logs/ts_dev_%j.log
# transfer_spacedust DEVELOPMENT variants (chosen after source207 numbers; labelled DEVELOPMENT everywhere):
# occurrence and expectation window w = 8 (gap <= 7, the engine's window on this collection) for both units, tau rule
# rerun on the w-8 shuffles; scored with every reading.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
step dev_ladder_w8 "$PY" -u "$C/words.py" ladder --specs syn,group --w 8 --threads "$T" --tag w8
TS8=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_w8.json'))['results']['syn']['tau_star'])")
TG8=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_w8.json'))['results']['group']['tau_star'])")
stamp "DEV w8 tau* syn $TS8 group $TG8"
step dev_score_P2w8 "$PY" -u "$C/wscore.py" --spec syn --tau "$TS8" --root "$TS/runs_w8" --tag DEV_P2w8 --compare "P2:V-coh0.1,P2:CC2" --threads "$T"
step dev_score_P1w8 "$PY" -u "$C/wscore.py" --spec group --tau "$TG8" --root "$TS/runs_w8" --tag DEV_P1w8 --compare "P1:V-coh0.1,P1:CC2" --threads "$T"
stamp "ALL DONE"
