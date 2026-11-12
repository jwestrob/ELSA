#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J cofold_split
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/split/log_%j.out
# Split-gene / fused-homolog check: the cognate chains (600 proteins) against all 3,585,303 DPANN proteins.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
C=$RD/cofold/split
cd $C
T=$SLURM_CPUS_ON_NODE
echo "threads $T host $(hostname) $(date)"
$MM createdb splitgene_queries.faa qdb
$MM createdb $RD/ws_dpann/work/proteins_gid.faa tdb --createdb-mode 1
$MM search qdb tdb aln tmp -s 7.5 -e 1e-3 --max-seqs 10000 --threads $T
$MM convertalis qdb tdb aln hits.tsv --format-output query,target,evalue,bits,pident,qstart,qend,qlen,tstart,tend,tlen --threads $T
wc -l hits.tsv
rm -rf tmp aln.*
echo "done $(date)"
