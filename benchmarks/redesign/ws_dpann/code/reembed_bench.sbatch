#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dpann_embbench
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/embbench_%j.log
# Timing benchmark and batch-invariance check for a clean ESM2-8M re-embedding of DPANN on CPU.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_ON_NODE"
echo "host $(hostname) cpus $SLURM_CPUS_ON_NODE start $(date -Is)"
/home/jwestrob/miniconda3/envs/elsa/bin/python $W/code/reembed_esm2.py --fasta $W/work/proteins_gid.faa \
    --out $W/package/embeddings/control --threads "$SLURM_CPUS_ON_NODE" --sample 5000
echo "end $(date -Is)"
