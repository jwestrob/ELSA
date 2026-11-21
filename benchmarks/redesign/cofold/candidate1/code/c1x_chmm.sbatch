#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J c1x_chmm
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/candidate1/logs/chmm_%j.log
# Candidate 1, third gene C: family HMM from the 258-row C instance alignment (deep dive, fs_c1/C_inst.aln.fa),
# searched against (1) all 3,585,303 DPANN proteins and (2) UniRef50 (2025 file). E <= 1e-3 reported; thresholds
# applied downstream.
set -euo pipefail
export PATH=/shared/software/bin:$PATH
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
DD=$RD/cofold/deepdive
X=$RD/cofold/candidate1
FAA=$RD/ws_dpann/work/proteins_gid.faa
T=$SLURM_CPUS_ON_NODE
H=$((T/2))
cd $X/hmm
echo "threads $T host $(hostname) $(date)"
hmmbuild --cpu $T --informat afa -n c1C c1C.hmm $DD/fs_c1/C_inst.aln.fa > hmmbuild_c1C.log
cat $DD/hmm/c1A.hmm $DD/hmm/c1B.hmm c1C.hmm > c1ABC.hmm
hmmsearch --cpu $H -E 1e-3 --domE 1e-3 --noali --domtblout c1C_vs_dpann.domtbl --tblout c1C_vs_dpann.tbl -o /dev/null c1C.hmm $FAA &
hmmsearch --cpu $H -E 1e-3 --domE 1e-3 --noali --domtblout c1C_vs_u50.domtbl --tblout c1C_vs_u50.tbl -o /dev/null \
  c1C.hmm /shared/db/uniref/uniref50/latest/uniref50.fasta &
wait
echo "done $(date)"
