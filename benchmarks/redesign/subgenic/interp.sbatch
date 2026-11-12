#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sg_interp
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/subgenic_interp_%j.log
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
"$RD/ws_tokens/env/bin/python" -u "$RD/subgenic/code/interp.py" "$RD/subgenic" "$RD/ws_dpann/package" \
   "$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet"
