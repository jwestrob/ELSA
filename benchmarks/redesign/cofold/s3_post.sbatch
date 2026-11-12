#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J cofold3_post
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s3/logs/post_%j.log
# Screen 3 refutation / recognition searches for the blocks with an interface passing both replicates:
#  1. per-chain PDB files of the cognate complexes (both replicates) -> Foldseek 10 vs PDB and AFDB Swiss-Prot
#     (foldseek DBs 'latest' 02_04_2026: PDB 2025-01-01, AFDB v6), -e 10, exhaustive prefilter off (screen-2 command)
#  2. MMseqs2 release 18 (-s 7.5, E <= 1e-3, --max-seqs 10000) of the same chains against all 3,585,303 DPANN proteins
#     (split-gene / fused-homolog check; screen-1/2 command)
# Inputs in $P: complexes.txt (one PDB path per line), splitgene_queries.faa (header = gid)
# Usage: sbatch --export=ALL,P=<dir> s3_post.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
DB=/shared/db/foldseek/latest/db
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T=$SLURM_CPUS_ON_NODE
cd $P
echo "threads $T host $(hostname) $(date) foldseek $($FS version)"
$PY $RD/cofold/s3/code/s2_split_chains.py $P $(cat complexes.txt)
FMT=query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,theader
for db in pdb alphafold_swissprot; do
  $FS easy-search chains $DB/$db ${db}.m8.tmp tmp_$db --threads $T -e 10 --format-output $FMT > log_$db.txt 2>&1
  mv ${db}.m8.tmp ${db}.m8; rm -rf tmp_$db
done
mkdir -p split && cd split
$MM createdb ../splitgene_queries.faa qdb
$MM createdb $RD/ws_dpann/work/proteins_gid.faa tdb --createdb-mode 1
$MM search qdb tdb aln tmp -s 7.5 -e 1e-3 --max-seqs 10000 --threads $T
$MM convertalis qdb tdb aln hits.tsv --format-output query,target,evalue,bits,pident,qstart,qend,qlen,tstart,tend,tlen --threads $T
wc -l hits.tsv
rm -rf tmp aln.* tdb*
echo "done $(date)"
