#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J gpufix_gate_b
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/gates/gate_b_%j.log

# Gate (b): CPU-path regression. The build under test (aksha 0.2.4.dev0) on the 4,000-protein testset3 sample,
# Pfam and KOfam with --cut_ga, CPU only, against production 0.2.3's CPU rows from job 119815.
set -uo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/gate_common.sh
echo "host $(hostname)  start $(date -u +%FT%TZ)  CPUs $SLURM_CPUS_ON_NODE"
aksha_dev --version
run_search b_pfam PFAM "$G/b_cpu_pfam"
run_search b_kofam KOFAM "$G/b_cpu_kofam"
compare_to b_pfam "$REF_PFAM" "$G/b_cpu_pfam/PFAM_hits_df.tsv"
compare_to b_kofam "$REF_KOFAM" "$G/b_cpu_kofam/KOFAM_hits_df.tsv"
echo "GATE_B_DONE $(date -u +%FT%TZ)"
