#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J tok_final
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/final_%j.log

# Recompute pair recall for every evaluated arm on the current label sets (adds tm_hi/tm_lo
# once tm_key exists), rebuild quality.tsv, and redo recall by identity/stratum for all arms.
set -euo pipefail
WS=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
cd "$WS/code"
echo "$(date -Is) host=$(hostname) threads=$T"
"$PY" -u quality.py recall --threads "$T"
"$PY" -u quality.py collect
"$PY" -u recall_strata.py --threads "$T"
echo "$(date -Is) END"
