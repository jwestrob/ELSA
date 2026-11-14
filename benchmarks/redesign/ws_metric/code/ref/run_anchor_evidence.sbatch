#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J elsa_anchorev
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/seq_invisible_20261111/logs/anchorev_%j.log

set -euo pipefail
RUN=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/seq_invisible_20261111
export OMP_NUM_THREADS="${SLURM_CPUS_ON_NODE:?}" MKL_NUM_THREADS="$SLURM_CPUS_ON_NODE" OPENBLAS_NUM_THREADS="$SLURM_CPUS_ON_NODE"
/home/jwestrob/miniconda3/envs/elsa/bin/python -u "$RUN/code/anchor_evidence.py" \
    --run "$RUN" --threads "$SLURM_CPUS_ON_NODE"
