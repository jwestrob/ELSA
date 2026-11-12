#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dd_hmm
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive/logs/hmm_%j.log
# (1) HMMs of the candidate-1 families (mafft alignments of token members) and Pfam 39.0 DUF7287/DUF7288/DUF7289
#     searched against all 3,585,303 DPANN proteins; (2) SignalP 6.0 (fast, organism other) on the cassette chains,
#     candidate-1 chains and the pilin-like control pool.
set -euo pipefail
export PATH=/shared/software/bin:$PATH
DD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive
FAA=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/work/proteins_gid.faa
T=$SLURM_CPUS_ON_NODE
cd $DD/hmm
echo "threads $T host $(hostname) $(date)"
hmmbuild --cpu $T --informat afa -n c1A c1A.hmm $DD/hh/c1A.aln.fa > hmmbuild_c1A.log
hmmbuild --cpu $T --informat afa -n c1B c1B.hmm $DD/hh/c1B.aln.fa > hmmbuild_c1B.log
cat c1A.hmm c1B.hmm > c1.hmm
printf "PF23958.3\nPF23959.3\nPF23960.4\n" > duf_keys.txt
hmmfetch -f /shared/db/pfam/39.0/Pfam-A.hmm duf_keys.txt > duf728x.hmm
hmmsearch --cpu $T -E 1e-3 --domE 1e-3 --noali --domtblout c1_vs_dpann.domtbl -o /dev/null c1.hmm $FAA &
hmmsearch --cpu $T --cut_ga --noali --domtblout duf728x_ga.domtbl -o /dev/null duf728x.hmm $FAA &
hmmsearch --cpu $T -E 1e-3 --domE 1e-3 --noali --domtblout duf728x_e3.domtbl -o /dev/null duf728x.hmm $FAA &
wait
echo "hmmsearch done $(date)"
mkdir -p $DD/sp/out
signalp --fastafile $DD/work/sp_in/signalp_in.fa --output_dir $DD/sp/out --format none --organism other \
  --mode fast --torch_num_threads $T --bsize 50
echo "done $(date)"
