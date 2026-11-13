#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sr4_py
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/softrec4_py_%j.log
# softrec4 steps that score readings (the scorer uses a process pool: BLAS threads 1). A step named ENGINE runs the
# token engine with --oriented on rescue3 tokens (control) into $RD/engine/results_dpann_softrec/dpann__softrec4__*.
# usage: export STEPS="ENGINE;readings4.py tune --configs ... --tag ..."; sbatch --export=ALL run4_py.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
PY=$RD/ws_tokens/env/bin/python
PYE=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/softrec/r4/tmp" NUMBA_CACHE_DIR="$RD/softrec/r4/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID"
IFS=';' read -r -a S <<< "${STEPS:?}"
for st in "${S[@]}"; do
    [[ -z "${st// }" ]] && continue
    echo "$(date -Is) STEP $st"
    if [[ "$st" == ENGINE ]]; then
        "$PYE" -u "$RD/engine/code/recurrence.py" --arm dpann__softrec4__tok_rescue3 --universe "$P/universe" \
            --tokens "$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet" --w 3 --null genome \
            --benchmark none --oriented --out "$RD/engine/results_dpann_softrec" --threads "$T"
        continue
    fi
    # shellcheck disable=SC2086
    "$PY" -u "$RD/softrec/r4/code/"$st --threads "$T"
done
echo "$(date -Is) done"
