#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sc2_triage
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/showcase2/logs/triage_%j.log
# ws_showcase2 enriched remote-homology triage (screen-2 commands): per family mafft -> HHblits UniRef30_2020_06
# (-n 2, -e 1e-3) -> HHsearch (enriched and DPANN-only) vs Pfam 39 HH and PDB100 2021-03-03; workers = CPUs // 4,
# 4 threads each; then (STEPS) hmmsearch Pfam 39 and RPS-BLAST CDD 3.20 on $SLURM_CPUS_ON_NODE threads.
# Two jobs may run 'families' together (per-family lock directories). Usage: sbatch --export=ALL,STEPS="families hmm cdd"
set -euo pipefail
export PATH=/shared/software/bin:$PATH
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
S=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/showcase2
cd $S
echo "threads $SLURM_CPUS_ON_NODE host $(hostname) $(date) steps ${STEPS:?}"
for st in $STEPS; do
  echo "STEP $st $(date)"
  $PY code/sc2_triage.py $st
done
echo "done $(date)"
