#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J hier_ml
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/hier_ml_%j.log
# Engine arms listed in $SETS (lines: arm<TAB>levels), PAR at a time, threads split evenly (total = allocation).
# REPRO=1 first runs the deployed v1.5 engine and the one-level ML engine on e3c50s75 and stops on any difference.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
C=$RD/hierarchy/code
T="${SLURM_CPUS_ON_NODE:?}"
PAR="${PAR:-4}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_THREADING_LAYER=workqueue TMPDIR="$RD/tmp"
echo "$(date -Is) host=$(hostname) cpus=$T job=$SLURM_JOB_ID sets=${SETS:?} par=$PAR"
if [[ "${REPRO:-0}" == "1" ]]; then
    E3=$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet
    export NUMBA_NUM_THREADS=$T NUMBA_CACHE_DIR=$RD/tmp/numba_${SLURM_JOB_ID}_v15
    mkdir -p "$NUMBA_CACHE_DIR"
    "$PY" -u "$RD/engine/code/recurrence.py" --arm dpann__mmseqs__seq__e3c50s75 --universe "$P/universe" --tokens "$E3" \
        --w 3 --null genome --benchmark none --all-genome-loci --out "$RD/engine/results_dpann_hier/v15check" --threads "$T"
    V=$RD/engine/results_dpann_hier/v15check/dpann__mmseqs__seq__e3c50s75__w3
    "$PY" -u $P/scorer/score_dpann_loci.py --pkg $P --loci $V/loci_all_genomes.parquet --out $V/score_min2 --threads $T --min-members 2 > $V/score_min2.log
    bash $C/one_ml.sh hier__single__e3c50s75 "e3c50s75=$E3" "$T"
    echo "== v1.5 (deployed) vs ML one level"
    "$PY" -u $C/compare_repro.py $V $RD/engine/results_dpann_hier/hier__single__e3c50s75__w3
    echo "== v1.5 rerun vs v1.1 base (information)"
    "$PY" -u $C/compare_repro.py $V $RD/engine/results_dpann_v2/dpann__mmseqs__seq__e3c50s75__w3 || true
fi
if [[ -n "${RIBO_LIST:-}" ]]; then        # standing ribosomal check on earlier result dirs (lines: result_dir [out_dir])
    export NUMBA_NUM_THREADS=$T NUMBA_CACHE_DIR=$RD/tmp/numba_${SLURM_JOB_ID}_ribo; mkdir -p "$NUMBA_CACHE_DIR"
    while read -r d o; do
        [[ -z "$d" ]] && continue
        "$PY" -u $RD/ribosomal/code/ribosomal_check.py check "$d" ${o:+--out "$o"} --threads $T 2>/dev/null < /dev/null || echo "RIBOSOMAL check failed: $d"
    done < "$RIBO_LIST"
fi
if [[ -n "${PRE_NESTED:-}" ]]; then "$PY" -u $C/nested.py $PRE_NESTED; fi    # nested levels of these sets first
TH=$(( T / PAR ))
grep -v '^#' "$SETS" | awk 'NF' | xargs -P "$PAR" -L 1 bash -c 'bash '"$C"'/'"${ONE:-one_ml.sh}"' "$0" "$1" '"$TH"
echo "$(date -Is) end"
