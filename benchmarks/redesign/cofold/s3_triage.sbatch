#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J cofold3_triage
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s3/logs/triage_%j.log
# Screen 3 enriched remote-homology triage (screen-2 search, s2_triage_remote.py 'families', via s3_triage.py):
# per family mafft -> HHblits UniRef30_2020_06 (-n 2, -e 1e-3) -> HHsearch (enriched and DPANN-only) vs Pfam 39 HH
# and PDB100 2021-03-03. Workers = CPUs // 4, 4 threads each.
set -euo pipefail
export PATH=/shared/software/bin:$PATH
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
S3=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s3
cd $S3
echo "threads $SLURM_CPUS_ON_NODE host $(hostname) $(date)"
$PY code/s3_triage.py families
echo "done $(date)"
