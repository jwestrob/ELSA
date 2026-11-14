#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dpann_storepool2
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/logs/storepool2_%j.log
# Second part of job 77588: spicy_lams (NCBI-protein store; sequences from sharur_ncbi.duckdb) and the SR-VP negative control.
# All Sharur paths are read-only inputs; outputs go to ws_dpann/package/embeddings/store_scope/.
set -euo pipefail
W=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann
S=/groups/banfield/users/jwestrob/bin/Sharur/data
E=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/srvp_research_2026-10-23/esm2
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
O=$W/package/embeddings/store_scope
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS="$SLURM_CPUS_ON_NODE" PYTHONDONTWRITEBYTECODE=1
N=$SLURM_CPUS_ON_NODE
echo "host $(hostname) cpus $N start $(date -Is)"
$PY $W/code/diagnose_store_pooling.py --name spicy_lams --h5 $S/spicy_lams/embeddings/protein_embeddings.h5 \
  --manifest $S/spicy_lams/embeddings/embedding_manifest.json --stage03 $S/spicy_lams/stage03_ncbi_proteins \
  --db $S/spicy_lams/sharur_ncbi.duckdb --windows 6 --threads $N --out $O
$PY $W/code/diagnose_store_pooling.py --name srvp_archaea --h5 $S/srvp_archaea/embeddings/protein_embeddings.h5 \
  --manifest $S/srvp_archaea/embeddings/embedding_manifest.json --stage03 $E/stage03_archaea \
  --db $S/srvp_archaea/sharur.duckdb --windows 6 --threads $N --out $O
$PY $W/code/faiss_readback.py \
  --index $S/srvp_archaea/embeddings/protein_embeddings.e43a2ec34cf9-20261023T180736-5ec4eaab.faiss \
  --ids $S/srvp_archaea/embeddings/protein_embeddings.e43a2ec34cf9-20261023T180736-5ec4eaab.ids.sqlite \
  --h5 $S/srvp_archaea/embeddings/protein_embeddings.h5 --rows-json $O/srvp_archaea_pooling.json \
  --out $O/srvp_archaea_faiss_readback.json
echo "end $(date -Is)"
