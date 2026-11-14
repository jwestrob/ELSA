#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ts_p2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/logs/ts_p2_%j.log
# transfer_spacedust P2: token cohesion, synteny-retention word ladder (obs + 3 shuffles), readings and source207 scores;
# then the token-matched engine reference rows (rescue3 tokens, w 3 and w 8).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
test -f "$TS/logs/done_zca" && test -f "$TS/logs/done_refs" && test -f "$TS/tok/mmseqs__seq__e3c50s75_rescue3__spacedust.parquet"
step cohesion "$PY" -u "$C/emb.py" cohesion --threads "$T"
step ladder_syn "$PY" -u "$C/words.py" ladder --specs syn --threads "$T" --tag syn
TAU=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_syn.json'))['results']['syn']['tau_star'])")
stamp "P2 tau* = $TAU"
step score_P2 "$PY" -u "$C/wscore.py" --spec syn --tau "$TAU" --tag P2 --threads "$T"
step engine_r3_w3 /home/jwestrob/miniconda3/envs/elsa/bin/python -u "$C/recurrence.py" --arm r3 \
    --tokens "$TS/tok/mmseqs__seq__e3c50s75_rescue3__spacedust.parquet" --w 3 --null genome --out "$TS/engine" --threads "$T"
step engine_r3_w8 /home/jwestrob/miniconda3/envs/elsa/bin/python -u "$C/recurrence.py" --arm r3 \
    --tokens "$TS/tok/mmseqs__seq__e3c50s75_rescue3__spacedust.parquet" --w 8 --null genome --out "$TS/engine" --threads "$T"
step ref_r3_w3 "$PY" -u "$C/refs.py" engine --rdir "$TS/engine/r3__w3" --name engine_r3_w3 --threads "$T"
step ref_r3_w8 "$PY" -u "$C/refs.py" engine --rdir "$TS/engine/r3__w8" --name engine_r3_w8 --threads "$T"
stamp "ALL DONE"
