#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J srvp_bins_callers_a
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bins/logs/callers_a_%j.log
# Workstream srvp_bins, callers part A of 2: DefenseFinder (defense-finder-models 3.1.0) and TXSScan through MacSyFinder
# 2.1.4 on shard 0/2 of the package genomes (dealt by descending gene count), $SLURM_CPUS_ON_NODE single-worker
# MacSyFinder processes. Raw MacSyFinder outputs only; part B (afterok) runs shard 1/2, writes gold/ and reseals.
# Estimated 20-28 CPU-h, 50-70 min on 24 CPUs.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/ws_srvp_bins
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T PKG=${PKG:?}"
rm -rf $W/work_callers
$PY -u $W/code/04_callers.py "$PKG" --work $W/work_callers --threads "$T" --shard 0/2 --run-only
stamp "DONE"
