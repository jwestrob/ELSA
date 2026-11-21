#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J slot_verifyA
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue/verify/logs/checkA_%j.log

# slot_rescue verify, check A (predeclared_verify.json): label models -> one HMM file (text copy), aksha search of the labeled
# proteins with no gathering cutoffs (--evalue 10 --domE 10, $SLURM_CPUS_ON_NODE threads), per-protein tables, null.
# SMOKE=N (sbatch --export=ALL,SMOKE=N) searches the first N proteins into A/smoke. The full run writes verify/STATUS_A.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
VD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue/verify
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
AKSHA=/home/jwestrob/.local/bin/aksha
N=${SMOKE:-0}
cd $VD
if [[ $N -gt 0 ]]; then OUT=$VD/A/smoke; ST=$OUT/STATUS; else OUT=$VD/A/full; ST=$VD/STATUS_A; fi
mkdir -p $OUT; rm -f $ST
echo "CPUs $SLURM_CPUS_ON_NODE host $(hostname) start $(date -u +%FT%TZ) smoke $N"
fail() { echo "FAILED $(date -u +%FT%TZ) job $SLURM_JOB_ID step $1" > $ST; cat $ST; exit 1; }
if [[ ! -s A/models/labels.hmm ]]; then $PY -u code/checkA.py models --out A/models || fail models; fi
FAA=$VD/data/labeled.faa
if [[ $N -gt 0 ]]; then FAA=$OUT/labeled_$N.faa; $PY -u code/checkA.py subset --n $N --out $FAA || fail subset; fi
$AKSHA --version
/usr/bin/time -v $AKSHA search --hmm_in A/models/labels.hmm --prot_in $FAA --outdir $OUT/aksha --evalue 10 --domE 10 \
    --threads "$SLURM_CPUS_ON_NODE" || fail aksha
[[ -s $OUT/aksha/user_hmms_hits_df.tsv ]] || fail no_hits_file
$PY -u code/checkA.py tables --hits $OUT/aksha/user_hmms_hits_df.tsv --out $OUT --n $N || fail tables
echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID smoke $N checkA_per_protein.parquet checkA_strata.tsv checkA_run.json in $OUT" > $ST
cat $ST
