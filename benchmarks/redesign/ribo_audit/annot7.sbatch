#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ribo7_annot
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit/work6/testset3/annot7_%j.log

# ribo_audit part 7, job 2: frozen annotation procedure (predeclared6.json; annot6.py unchanged, sha256 7eb20f2b...68ad)
# through Aksha with $SLURM_CPUS_ON_NODE threads; ribosomal tables from job 1; N_total = package protein count.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 AKSHA=/home/jwestrob/.local/bin/aksha
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/testset3/package
cd $RD/ribo_audit
NPROT=$(grep -c '^>' $P/proteins_gid.faa)
echo "CPUs $SLURM_CPUS_ON_NODE proteins $NPROT"
/usr/bin/time -v /home/jwestrob/miniconda3/envs/elsa/bin/python -u code/annot6.py run --faa $P/proteins_gid.faa \
    --genes $P/universe/genes.parquet --contigs $P/universe/contigs.parquet \
    --marked work6/testset3/ribosomal/marked.parquet --runs work6/testset3/ribosomal/runs.parquet \
    --out work6/testset3/annot --z_total "$NPROT" --threads "$SLURM_CPUS_ON_NODE"
