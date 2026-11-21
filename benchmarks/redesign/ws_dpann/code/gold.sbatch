#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dpann_gold
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/gold_%j.log
# Gold loci from curated callers, per-protein annotation flags, then the scorer self-test.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
echo "host $(hostname) cpus $SLURM_CPUS_ON_NODE start $(date -Is)"
$PY $W/code/build_dpann_package.py --db $W/input/sharur.duckdb --seal $W/input/dataset.seal.json \
    --pkg $W/package --work $W/work --threads "$SLURM_CPUS_ON_NODE" gold
mkdir -p $W/package/scorer
$PY $W/code/selftest_scorer.py --pkg $W/package --threads "$SLURM_CPUS_ON_NODE" --out $W/package/scorer/selftest.json
echo "end $(date -Is)"
