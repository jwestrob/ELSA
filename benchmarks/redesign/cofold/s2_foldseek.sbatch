#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J cofold2_fs
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s2/logs/foldseek_%j.log
# Foldseek 10 of predicted chains (QDIR/chains) against PDB and AFDB Swiss-Prot (foldseek DBs 'latest',
# 02_04_2026: PDB 2025-01-01, AFDB v6 2025_03) and multimer search of the complexes (QDIR/complex) against PDB.
# Usage: sbatch --export=ALL,QDIR=<dir> s2_foldseek.sbatch
set -euo pipefail
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
DB=/shared/db/foldseek/latest/db
T=$SLURM_CPUS_ON_NODE
cd $QDIR
echo "threads $T host $(hostname) $(date) foldseek $($FS version)"
FMT=query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,theader
for db in pdb alphafold_swissprot; do
  if [ ! -s ${db}.m8 ]; then
    $FS easy-search chains $DB/$db ${db}.m8.tmp tmp_$db --threads $T -e 10 --format-output $FMT > log_$db.txt 2>&1
    mv ${db}.m8.tmp ${db}.m8; rm -rf tmp_$db
  fi
done
# multimer search aborts in expandmultimer (std::out_of_range, job 78255); complexes are summarised from
# per-chain hits to the same PDB entry (s2_foldseek_summary.py); one non-fatal attempt with Foldseek 9 is kept
if [ ! -s pdb_multimer.m8 ]; then
  /shared/software/foldseek/9-427df8a/bin/foldseek easy-multimersearch complex $DB/pdb pdb_multimer.m8.tmp tmp_mm --threads $T -e 10 > log_mm.txt 2>&1 \
    && mv pdb_multimer.m8.tmp pdb_multimer.m8 || echo "multimer search failed (non-fatal)"
  rm -rf tmp_mm
fi
ls -la; echo "done $(date)"
