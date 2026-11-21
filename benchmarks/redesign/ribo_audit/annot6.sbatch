#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ribo6_annot
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit/work6/annot6_%j.log

# ribo_audit part 6: frozen annotation procedure (predeclared6.json) through Aksha for a sealed package.
# TEMPLATE: submit only after the coordinator states that $RD/testset3 is sealed. Usage:
#   sbatch --export=ALL,PKG=<package root>,OUT=<output dir under ribo_audit/work6> annot6.sbatch
# PKG must hold universe/{genes,contigs}.parquet, a gid-headed protein FASTA (universe/proteins.faa or proteins_gid.faa)
# and the ribosomal tables (ribosomal/{marked,runs}.parquet or gold/ribosomal_{marked,runs}.parquet).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 AKSHA=/home/jwestrob/.local/bin/aksha
cd /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
FAA=$PKG/universe/proteins.faa; [ -f "$FAA" ] || FAA=$PKG/proteins_gid.faa
MK=$PKG/ribosomal/marked.parquet; [ -f "$MK" ] || MK=$PKG/gold/ribosomal_marked.parquet
RN=$PKG/ribosomal/runs.parquet; [ -f "$RN" ] || RN=$PKG/gold/ribosomal_runs.parquet
NPROT=$(grep -c '^>' "$FAA")
echo "CPUs $SLURM_CPUS_ON_NODE proteins $NPROT"
$PY -u code/annot6.py run --faa "$FAA" --genes $PKG/universe/genes.parquet --contigs $PKG/universe/contigs.parquet \
    --marked "$MK" --runs "$RN" --out "$OUT" --z_total "$NPROT" --threads "$SLURM_CPUS_ON_NODE"
