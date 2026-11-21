#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J large_loci_fam2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/large_loci_dpann/logs/fam2_%j.log

# large_loci_dpann: Z lists for genes of large outermost loci (+ shuffled null genes), locus links, families,
# gene clusters (code/families.py, predeclared.json). Pool of $SLURM_CPUS_ON_NODE workers. SMOKE=N runs N loci.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/large_loci_dpann
cd $W
mkdir -p out2; rm -f out2/STATUS
echo "CPUs $SLURM_CPUS_ON_NODE host $(hostname) start $(date -u +%FT%TZ) smoke ${SMOKE:-0}"
/usr/bin/time -v /home/jwestrob/miniconda3/envs/elsa/bin/python -u code/families2.py --threads "$SLURM_CPUS_ON_NODE"
rc=$?
if [[ $rc -eq 0 ]]; then echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID smoke ${SMOKE:-0}" > out2/STATUS
else echo "FAILED $(date -u +%FT%TZ) job $SLURM_JOB_ID rc $rc" > out2/STATUS; fi
cat out2/STATUS
