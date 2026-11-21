#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J bs_match
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_refseq_synton/logs/bs_match_%j.log
# baseline_synton task 4: gembase genome code -> RefSeq assembly mapping (DuckDB join; no method run, nothing scored).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/ws_refseq_synton
T="${SLURM_CPUS_ON_NODE:?}"
export PATH="$RD/baseline_synton/env/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
mkdir -p "$W/tmp_duck"
/usr/bin/time -v -o "$W/logs/match_${SLURM_JOB_ID}.time" python -u "$RD/baseline_synton/code/match_gembase.py" --w "$W" --threads "$T"
rm -rf "$W/tmp_duck"
