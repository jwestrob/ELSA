#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J sg_analyze
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/subgenic_analyze_%j.log
# Sub-genic sizing, job 3: analysis of hits/S, B, C and rep/ (DuckDB and numba on all allocated CPUs).
# ASTAGES (comma list, exported in the environment) selects stages; default all.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/subgenic
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export NUMBA_NUM_THREADS=$T OMP_NUM_THREADS=1
echo "$(date -u +%FT%TZ) host=$(hostname) threads=$T stages=${ASTAGES:-all}"
"$PY" -u "$W/code/analyze.py" --work "$W" --pkg "$RD/ws_dpann/package" \
   --tokens "$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet" \
   --scores "$RD/engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3/score_min2/instance_scores.parquet" \
   --stages "${ASTAGES:-composite,fragment,tokens,gold,giants}" --max-seqs 8000 --db "${ADB:-analyze.duckdb}"
echo "$(date -u +%FT%TZ) done"
