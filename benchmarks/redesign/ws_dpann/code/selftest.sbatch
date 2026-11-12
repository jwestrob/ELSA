#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dpann_selftest
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/selftest_%j.log
# Scorer self-test against the final scorer version, plus a whole-contig example run of the CLI.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
echo "host $(hostname) cpus $SLURM_CPUS_ON_NODE start $(date -Is)"
mkdir -p $W/package/scorer
cp $W/code/score_dpann_loci.py $W/code/selftest_scorer.py $W/package/scorer/
$PY $W/package/scorer/selftest_scorer.py --pkg $W/package --threads "$SLURM_CPUS_ON_NODE" --out $W/package/scorer/selftest.json
# CLI example: whole-contig loci (the trivial upper bound on coverage)
$PY - <<PYEOF
import pandas as pd
c = pd.read_parquet("$W/package/universe/contigs.parquet")
pd.DataFrame(dict(genome=c.genome, contig=c.contig, start=0, end=c.n_genes - 1, module_id=c.contig_code)).to_parquet(
    "$W/work/example_whole_contig_loci.parquet", index=False)
PYEOF
$PY $W/package/scorer/score_dpann_loci.py --pkg $W/package --loci $W/work/example_whole_contig_loci.parquet \
    --out $W/package/scorer/example_whole_contig --threads "$SLURM_CPUS_ON_NODE"
echo "end $(date -Is)"
