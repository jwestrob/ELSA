#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J ribo7_mark
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ribo_audit/work6/testset3/mark7_%j.log

# ribo_audit part 7, job 1: ribosomal marking and runs for testset3 by the package rule (03_build_package.py st_ribosomal),
# search through Aksha (--hmm_in ribo.hmm --cut_ga, $SLURM_CPUS_ON_NODE threads); tables written outside the sealed package;
# RUNS_READY.json written at the end.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 AKSHA=/home/jwestrob/.local/bin/aksha
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
cd $RD/ribo_audit
echo "CPUs $SLURM_CPUS_ON_NODE"
/usr/bin/time -v /home/jwestrob/miniconda3/envs/elsa/bin/python -u code/mark7.py run --pkg $RD/testset3/package \
    --faa $RD/testset3/package/proteins_gid.faa --out work6/testset3 --threads "$SLURM_CPUS_ON_NODE" \
    --ready work6/testset3/RUNS_READY.json
