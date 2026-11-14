#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dpann_build_b
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/build_b_%j.log
# Stage B: March ESM2-8M vectors aligned to the canonical proteins, then the re-embedding control.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_ON_NODE"
echo "host $(hostname) cpus $SLURM_CPUS_ON_NODE start $(date -Is)"
$PY $W/code/build_dpann_package.py --db $W/input/sharur.duckdb --seal $W/input/dataset.seal.json \
    --pkg $W/package --work $W/work --threads "$SLURM_CPUS_ON_NODE" embed
$PY $W/code/reembed_control.py --pkg $W/package --work $W/work --db $W/input/sharur.duckdb \
    --threads "$SLURM_CPUS_ON_NODE" --out $W/package/embeddings/control
echo "end $(date -Is)"
