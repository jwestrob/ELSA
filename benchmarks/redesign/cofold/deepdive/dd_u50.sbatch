#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dd_u50
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive/logs/u50_%j.log
# Candidate-1 family HMMs (A, B, and the concatenated A|B instance alignment) against UniRef50 (2025 release file,
# /shared/db/uniref/uniref50/latest/uniref50.fasta). Substitute for BFD, whose a3m/hhm files on /shared are not
# readable (mode 600, owner gleno).
set -euo pipefail
export PATH=/shared/software/bin:$PATH
DD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive
T=$SLURM_CPUS_ON_NODE
cd $DD/hmm
echo "threads $T host $(hostname) $(date)"
hmmbuild --cpu $T --informat afa -n c1AB c1AB.hmm $DD/hh/c1AB.aln.fa > hmmbuild_c1AB.log
cat c1A.hmm c1B.hmm c1AB.hmm > c1_all.hmm
hmmsearch --cpu $T -E 1e-2 --domE 1e-2 --noali --domtblout u50_c1.domtbl --tblout u50_c1.tbl -o /dev/null \
  c1_all.hmm /shared/db/uniref/uniref50/latest/uniref50.fasta
echo "done $(date)"
