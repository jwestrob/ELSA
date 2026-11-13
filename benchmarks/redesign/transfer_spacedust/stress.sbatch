#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ts_stress
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/logs/ts_stress_%j.log
# transfer_spacedust SECONDARY: duplicate-genome stress test (amendment_0). Runs are sequential; numba kernels use the
# allocation's threads (synteny N_eff) and the group linkage uses BLAS threads.
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/code/env.sh
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
TAU=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_group_genome.json'))['results']['group']['tau_star'])")
TAUG=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_group_genome.json'))['results']['genome']['tau_star'])")
TAUS=$("$PY" -c "import json; print(json.load(open('$TS/results/ladder_syn.json'))['results']['syn']['tau_star'])")
stamp "taus genome $TAUG group $TAU syn $TAUS"
step stress "$PY" -u "$C/stress.py" --taus "genome:$TAUG,group:$TAU,syn:$TAUS" --threads "$T"
stamp "ALL DONE"
