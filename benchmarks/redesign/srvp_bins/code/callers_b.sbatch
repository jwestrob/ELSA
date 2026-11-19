#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J srvp_bins_callers_b
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bins/logs/callers_b_%j.log
# Workstream srvp_bins, callers part B of 2 (submitted with --dependency=afterok on part A): shard 1/2, then one parse
# of every genome's MacSyFinder output (--reuse; any genome without output is computed here) into gold/instances.parquet
# and gold/members.parquet, README regeneration and the package seal. Estimated 20-28 CPU-h, 50-70 min on 24 CPUs.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/ws_srvp_bins
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T PKG=${PKG:?}"
$PY -u $W/code/04_callers.py "$PKG" --work $W/work_callers --threads "$T" --shard 1/2 --run-only
$PY -u $W/code/04_callers.py "$PKG" --work $W/work_callers --threads "$T" --reuse
rm -f "$PKG/code/callers.sbatch"
cp $W/code/04_callers.py $W/code/callers_a.sbatch $W/code/callers_b.sbatch $W/code/05_readme.py $W/code/03_build_package.py "$PKG/code/"
$PY $W/code/05_readme.py "$PKG" > /dev/null
$PY -u $W/code/03_build_package.py "$PKG/selection.tsv" --pkg "$PKG" --work $W/work_pkg --threads "$T" seal
rm -rf $W/work_callers/faa
stamp "DONE"
