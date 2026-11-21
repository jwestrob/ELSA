#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J c1x_nbscan
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/candidate1/logs/nbscan_%j.log
# Task 1: every CDS of the 40 public nucleotide records holding a bacterial / non-DPANN A, B or fused A|B homolog
# (public sequences, fetched from NCBI by accession) against Pfam 39.0 (hmmscan, GA cutoffs and E <= 1e-3 both
# reported) and against the candidate-1 family HMMs A, B (deep dive) and C (job 78501), all hits E <= 10.
set -euo pipefail
export PATH=/shared/software/bin:$PATH
X=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/candidate1
T=$SLURM_CPUS_ON_NODE
cd $X/nb
echo "threads $T host $(hostname) $(date)"
hmmsearch --cpu 4 -E 10 --domE 10 --noali --domtblout nb_c1abc.domtbl -o /dev/null $X/hmm/c1ABC.hmm contig_proteins.faa &
hmmscan --cpu $((T-4)) -E 1e-3 --domE 1e-3 --noali --domtblout nb_pfam.domtbl -o /dev/null /shared/db/pfam/39.0/Pfam-A.hmm contig_proteins.faa
wait
echo "done $(date)"
