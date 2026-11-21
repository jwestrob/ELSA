#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dpann_reembed
#SBATCH --array=0-7%2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/reembed_%A_%a.log
# NOT SUBMITTED (awaiting the coordinator's decision). Clean ESM2-8M re-embedding of all DPANN proteins,
# 8 shards by gid % 8, two at a time (dpann budget). Benchmark: ~33 h on 24 CPUs in total, ~4.2 h per shard.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_ON_NODE"
echo "host $(hostname) cpus $SLURM_CPUS_ON_NODE task $SLURM_ARRAY_TASK_ID start $(date -Is)"
/home/jwestrob/miniconda3/envs/elsa/bin/python $W/code/reembed_esm2.py --fasta $W/work/proteins_gid.faa \
    --out $W/work/reembed_esm2_8m --threads "$SLURM_CPUS_ON_NODE" --shard "$SLURM_ARRAY_TASK_ID/8"
echo "end $(date -Is)"
