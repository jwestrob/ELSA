#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J cas_big
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/softrec_cas_big_%j.log
# ws_cas steps (diag_cas.py / fix_cas.py subcommands); --threads is appended. Steps whose name contains "repro",
# "part1" or "score" run the scorer's process pool: BLAS threads 1 for those.
# usage: export STEPS="diag_cas.py repro;diag_cas.py part1"; sbatch --export=ALL run_cas_big.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/softrec/cas/tmp" NUMBA_CACHE_DIR="$RD/softrec/cas/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID"
IFS=';' read -r -a S <<< "${STEPS:?}"
for st in "${S[@]}"; do
    [[ -z "${st// }" ]] && continue
    echo "$(date -Is) STEP $st"
    if [[ "$st" == *repro* || "$st" == *part1* || "$st" == *score* || "$st" == *attrib* || "$st" == *eval* || "$st" == *fix_cas* ]]; then B=1; else B=$T; fi
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=$B OPENBLAS_NUM_THREADS=$B MKL_NUM_THREADS=$B "$PY" -u "$RD/softrec/cas/code/"$st --threads "$T"
done
echo "$(date -Is) done"
