#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sr_soft
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/softrec_soft_%j.log
# Token-free recurrence runs, then the DPANN scorer, paired comparisons and the ribosomal check of every
# result directory they wrote.
# usage: export RUNS="<softrec args>;<softrec args>;..."  (each must carry --name NAME), then
#        sbatch --export=ALL run_soft.sbatch   (the 64-CPU variant: sbatch --qos=standard_expand_2 -c 64 ...)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/tmp" NUMBA_CACHE_DIR="$RD/tmp/numba_softrec_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID"
IFS=';' read -r -a R <<< "${RUNS:?}"
for args in "${R[@]}"; do
    [[ -z "${args// }" ]] && continue
    name=$(sed -E 's/.*--name ([^ ]+).*/\1/' <<< "$args")
    echo "$(date -Is) RUN [$name] softrec.py $args"
    # shellcheck disable=SC2086
    "$PY" -u "$RD/softrec/code/softrec.py" $args --threads "$T"
    dirs=( "$RD"/engine/results_dpann_softrec/dpann__softrec__"${name}"__* )
    bash "$RD/softrec/code/score_dirs.sh" "$T" "${dirs[@]}"
done
echo "$(date -Is) done"
