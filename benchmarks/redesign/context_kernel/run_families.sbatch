#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J ctxk_fam
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/ctxk_fam_%j.log

# Context kernel, stage 2: Leiden (modularity) on candidate or supported edges, one process per
# (weight, resolution). Leiden is single-threaded; PROCS processes share the graph by fork.
# env: EDGE_SET=all|sup  VARIANT=primary|secondary|<name>  WEIGHTS=sim,simctx  RES=<comma list>  [PROCS]
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$RD/tmp"
"$PY" -u "$RD/context_kernel/code/context_kernel.py" families --edge-set "${EDGE_SET:?}" --variant "${VARIANT:-primary}" \
    --weights "${WEIGHTS:-sim,simctx}" --res "${RES:?}" --procs "${PROCS:-$T}" --seed 17
