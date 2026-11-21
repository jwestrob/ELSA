#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J f17_fs
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine/logs/fs_%j.log
# f17_machine: local Foldseek of ESMFold2 monomers vs PDB and AFDB-SwissProt (as large_loci_srvp s09); structures stay on biotite.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine
DB=/shared/db/foldseek/latest/db
FS=/shared/software/bin/foldseek
cd $W/folds
FMT=query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,prob,alntmscore,qtmscore,ttmscore,theader
for db in pdb alphafold_swissprot; do
  $FS easy-search fsq $DB/$db fs_$db.tsv tmp_$db --threads "$SLURM_CPUS_ON_NODE" -e 0.01 --max-seqs 1000 --format-output $FMT > fslog_$db.txt 2>&1
  rm -rf tmp_$db
done
echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID" > FS_STATUS
