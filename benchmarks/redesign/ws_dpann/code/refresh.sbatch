#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dpann_refresh
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/refresh_%j.log
# Rebuild genomes.parquet and gold tables (integer counts, src_* flag names), rerun the self-test and the CLI example.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
export PYTHONDONTWRITEBYTECODE=1
echo "host $(hostname) cpus $SLURM_CPUS_ON_NODE start $(date -Is)"
$PY $W/code/build_dpann_package.py --db $W/input/sharur.duckdb --seal $W/input/dataset.seal.json \
    --pkg $W/package --work $W/work --threads "$SLURM_CPUS_ON_NODE" taxonomy gold
rm -rf $W/package/scorer/__pycache__ $W/package/scorer/example_whole_contig
cp $W/code/score_dpann_loci.py $W/code/selftest_scorer.py $W/package/scorer/
$PY $W/package/scorer/selftest_scorer.py --pkg $W/package --threads "$SLURM_CPUS_ON_NODE" --out $W/package/scorer/selftest.json
$PY $W/package/scorer/score_dpann_loci.py --pkg $W/package --loci $W/work/example_whole_contig_loci.parquet \
    --out $W/package/scorer/example_whole_contig --threads "$SLURM_CPUS_ON_NODE"
echo "end $(date -Is)"
