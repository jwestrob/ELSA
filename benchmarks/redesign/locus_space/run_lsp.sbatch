#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J lsp
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/locus_space/logs/lsp_%j.log
# Usage: sbatch --export=ALL,SCRIPT=lspace.py,STAGE=gold run_lsp.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE OPENBLAS_NUM_THREADS=$SLURM_CPUS_ON_NODE MKL_NUM_THREADS=$SLURM_CPUS_ON_NODE
export NUMBA_NUM_THREADS=$SLURM_CPUS_ON_NODE NUMBA_CACHE_DIR=$RD/locus_space/tmp/numba_cache
export OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
mkdir -p $NUMBA_CACHE_DIR
cd $RD/locus_space/code
echo "host $(hostname) cpus $SLURM_CPUS_ON_NODE script ${SCRIPT:-lspace.py} stage $STAGE start $(date -u)"
/usr/bin/time -v $PY ${SCRIPT:-lspace.py} $STAGE --threads $SLURM_CPUS_ON_NODE ${EXTRA:-}
echo "end $(date -u)"
