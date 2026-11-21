#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J f17_mm
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine/logs/mm_%j.log
# f17_machine: (1) DPANN proteome from the biotite copy of the Sharur DB (read-only); (2) MMseqs2 search of family-17
# role sequences (<=250 per role) against it, to map each role's homologs in every DPANN genome; (3) TMHMM 2.0c on all
# role members; (4) SignalP 6 (fast, organism other) on the role subsets.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/dpann_leads/f17_machine
DB=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/input/sharur.duckdb
cd $W/mm
/home/jwestrob/miniconda3/envs/elsa/bin/python - <<PY
import duckdb
c = duckdb.connect('$DB', read_only=True)
with open('dpann.faa', 'w') as fh:
    for pid, s in c.sql("select protein_id, rtrim(sequence, '*') from proteins").fetchall():
        fh.write(f'>{pid}\n{s}\n')
PY
for f in $W/roles/*.sub.faa; do g=$(basename $f .sub.faa); sed "s/^>/>${g}|/" $f; done > queries.faa
/shared/software/bin/mmseqs easy-search queries.faa dpann.faa hits.m8 tmp -s 7.5 -e 1e-5 --max-seqs 5000 --threads $SLURM_CPUS_ON_NODE \
   --format-output query,target,pident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits > mm.log 2>&1
rm -rf tmp
cd $W/topo
cat $W/roles/*.all.faa > all_roles.faa
/shared/software/bin/tmhmm -short all_roles.faa > tmhmm.tsv 2> tmhmm.err || true
cat $W/roles/*.sub.faa | awk '/^>/{print; next}{print}' > sub_roles.faa
signalp --fastafile sub_roles.faa --output_dir signalp --format none --organism other --mode fast > signalp.log 2>&1 || true
echo "DONE $(date -u +%FT%TZ) job $SLURM_JOB_ID" > $W/mm/STATUS
