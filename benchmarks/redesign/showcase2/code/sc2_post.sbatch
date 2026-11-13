#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sc2_post
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/showcase2/logs/post_%j.log
# ws_showcase2 refutation / recognition searches for blocks with an interface passing both replicates (census commands,
# s3_post.sbatch, unchanged in their parameters):
#  1. per-chain PDB files of the cognate complexes (both replicates) -> Foldseek 10 vs PDB and AFDB Swiss-Prot
#     (/shared/db/foldseek/latest/db), -e 10, all threads
#  2. MMseqs2 release 18 (-s 7.5, E <= 1e-3, --max-seqs 10000) of the same chains against all 3,585,303 DPANN proteins
#     (split-gene / fused-homolog check)
#  3. US-align (multimer, -mm 1 -ter 0) of replicate 1 vs replicate 2 per block, run in parallel over blocks
# Inputs in $P: complexes.txt (one PDB path per line), splitgene_queries.faa (header = gid), usalign_pairs.txt (r1 r2)
# Usage: sbatch --export=ALL,P=<dir> sc2_post.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
DB=/shared/db/foldseek/latest/db
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
US=/shared/software/usalign/latest/bin/USalign
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T=$SLURM_CPUS_ON_NODE
cd $P
echo "threads $T host $(hostname) $(date) foldseek $($FS version) usalign $US"
$PY $RD/cofold/s3/code/s2_split_chains.py $P $(cat complexes.txt)
FMT=query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,theader
for db in pdb alphafold_swissprot; do
  $FS easy-search chains $DB/$db ${db}.m8.tmp tmp_$db --threads $T -e 10 --format-output $FMT > log_$db.txt 2>&1
  mv ${db}.m8.tmp ${db}.m8; rm -rf tmp_$db
done
# US-align replicate complexes, T in parallel
mkdir -p us
cat usalign_pairs.txt | xargs -P $T -n 2 sh -c "$US \$0 \$1 -mm 1 -ter 0 -outfmt 2 > us/\$(basename \$0 .pdb)__\$(basename \$1 .pdb).txt 2>&1"
mkdir -p split && cd split
$MM createdb ../splitgene_queries.faa qdb
$MM createdb $RD/ws_dpann/work/proteins_gid.faa tdb --createdb-mode 1
$MM search qdb tdb aln tmp -s 7.5 -e 1e-3 --max-seqs 10000 --threads $T
$MM convertalis qdb tdb aln hits.tsv --format-output query,target,evalue,bits,pident,qstart,qend,qlen,tstart,tend,tlen --threads $T
wc -l hits.tsv
rm -rf tmp aln.* tdb*
echo "done $(date)"
