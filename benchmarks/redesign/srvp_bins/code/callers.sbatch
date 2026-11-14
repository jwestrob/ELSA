#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J srvp_bins_callers
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bins/logs/callers_%j.log
# Workstream srvp_bins: DefenseFinder (defense-finder-models 3.1.0) and TXSScan through MacSyFinder 2.1.4 on every
# package genome, $SLURM_CPUS_ON_NODE single-worker MacSyFinder processes; then the package seal is rewritten so it covers
# gold/. Estimated 41-47 CPU-h (pilot: 143 s per 5,270-protein genome for the DefenseFinder family alone).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/ws_srvp_bins
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T PKG=${PKG:?}"
rm -rf $W/work_callers
$PY -u $W/code/04_callers.py "$PKG" --work $W/work_callers --threads "$T"
cp $W/code/04_callers.py $W/code/callers.sbatch $W/code/05_readme.py "$PKG/code/"
$PY $W/code/05_readme.py "$PKG" > /dev/null
$PY -u $W/code/03_build_package.py "$PKG/selection.tsv" --pkg "$PKG" --work $W/work_pkg --threads "$T" seal
rm -rf $W/work_callers/faa
stamp "DONE"
