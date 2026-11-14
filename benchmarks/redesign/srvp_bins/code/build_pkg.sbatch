#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J srvp_bins_pkg
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bins/logs/build_pkg_%j.log
# Workstream srvp_bins: package of the dereplicated primary tier (env SEL, PKG, WORK). Stages: genes (GTDB-Tk Prodigal
# calls for representatives, Prodigal with GTDB-Tk's procedure for other picks and 6 controls; $SLURM_CPUS_ON_NODE
# processes), universe, fasta, ribosomal ($SLURM_CPUS_ON_NODE single-threaded hmmsearch processes), split, seal.
# No ELSA method, no embeddings.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/ws_srvp_bins
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T SEL=${SEL:?} PKG=${PKG:?} WORK=${WORK:?}"
rm -rf "$PKG" "$WORK"
$PY -u $W/code/03_build_package.py "$SEL" --pkg "$PKG" --work "$WORK" --threads "$T" genes universe fasta ribosomal split
mkdir -p "$PKG/code"; cp $W/code/02_select.py $W/code/03_build_package.py $W/code/build_pkg.sbatch "$PKG/code/"
cp "$SEL" "$PKG/selection.tsv"
rm -rf "$WORK/duck_tmp" "$WORK/prodigal_tmp"
$PY -u $W/code/03_build_package.py "$SEL" --pkg "$PKG" --work "$WORK" --threads "$T" seal
stamp "DONE"
