#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J slot_rescue_pass
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue/logs/pass_%j.log

# slot_rescue whole-DPANN pass (predeclared_pass.json): regression gate, Z slot call on every gene without a Pfam row,
# calibration on 50,000 bonded Pfam genes, A support column, KO cross-check, tables. Pool of $SLURM_CPUS_ON_NODE workers.
# Writes pass/* and pass/STATUS (DONE or FAILED). SMOKE=N (sbatch --export=ALL,SMOKE=N) runs N items per stage.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SRD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue
cd $SRD
mkdir -p pass
rm -f pass/STATUS pass/stop_reason.txt
echo "CPUs $SLURM_CPUS_ON_NODE host $(hostname) start $(date -u +%FT%TZ) smoke ${SMOKE:-0}"
/usr/bin/time -v /home/jwestrob/miniconda3/envs/elsa/bin/python -u code/pass_run.py --threads "$SLURM_CPUS_ON_NODE" --smoke "${SMOKE:-0}"
rc=$?
if [[ $rc -eq 0 && -s pass/dpann_slot_rescue_pass.parquet ]]; then
  echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID smoke ${SMOKE:-0} dpann_slot_rescue_pass.parquet counts.tsv calibration_cells.tsv regression.json run.json" > pass/STATUS
else
  echo "FAILED $(date -u +%FT%TZ) job $SLURM_JOB_ID rc $rc $(cat pass/stop_reason.txt 2>/dev/null | head -c 300)" > pass/STATUS
fi
cat pass/STATUS
