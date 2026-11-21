#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ts3_select
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/testset3/logs/select_%j.log
# testset3 selection from GTDB r232 metadata only (DuckDB over the metadata TSVs, $SLURM_CPUS_ON_NODE threads).
set -euo pipefail
T3=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/testset3
PY=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/env/bin/python
echo "$(date -u +%FT%TZ) host $(hostname) cpus $SLURM_CPUS_ON_NODE"
$PY -u $T3/code/ts3_select.py $T3/selection "$SLURM_CPUS_ON_NODE" $T3/selection_rules.txt
echo "$(date -u +%FT%TZ) done"
