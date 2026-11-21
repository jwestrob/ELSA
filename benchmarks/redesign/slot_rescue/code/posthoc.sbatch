#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J slot_rescue_ph
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue/logs/posthoc_%j.log

# slot_rescue POST HOC diagnostic (not predeclared): whole-list votes with and without the positional filter.
# Same lists / labels / vote rule as score.py; Pool of $SLURM_CPUS_ON_NODE workers.
# Writes results/posthoc.parquet and rewrites STATUS (DONE or FAILED).
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SRD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue
cd $SRD
rm -f STATUS
echo "CPUs $SLURM_CPUS_ON_NODE host $(hostname) start $(date -u +%FT%TZ)"
/usr/bin/time -v /home/jwestrob/miniconda3/envs/elsa/bin/python -u code/posthoc.py --threads "$SLURM_CPUS_ON_NODE"
rc=$?
if [[ $rc -eq 0 && -s results/posthoc.parquet ]]; then
  echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID posthoc.parquet (after job 119461 calls.parquet gate.json run.json)" > STATUS
else
  echo "FAILED $(date -u +%FT%TZ) job $SLURM_JOB_ID rc $rc (see logs/posthoc_${SLURM_JOB_ID}.log)" > STATUS
fi
cat STATUS
