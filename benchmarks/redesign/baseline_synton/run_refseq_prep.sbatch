#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J bs_refseq
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_refseq_synton/logs/bs_refseq_%j.log
# baseline_synton task 4: inventory of the Zenodo tables and the universe of the RefSeq May-2023 complete-genome cohort.
# No method is run, nothing is clustered or scored.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/ws_refseq_synton; C=$RD/baseline_synton/code
T="${SLURM_CPUS_ON_NODE:?}"
export PATH="$RD/baseline_synton/env/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
mkdir -p "$W/tmp_duck"
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
if [[ "${INVENTORY:-1}" == 1 ]]; then
  /usr/bin/time -v -o "$W/logs/inventory_${SLURM_JOB_ID}.time" python -u "$C/zenodo_inventory.py" --w "$W" --threads "$T"
fi
if [[ "${UNIVERSE:-1}" == 1 ]]; then
  /usr/bin/time -v -o "$W/logs/universe_${SLURM_JOB_ID}.time" python -u "$C/build_refseq_universe.py" --w "$W" --threads "$T"
fi
rm -rf "$W/tmp_duck"
stamp "DONE"
