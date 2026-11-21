#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J gr_eval
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/grammar_eval_%j.log
# usage: STEPS="e123 loci score learned" sbatch --export=ALL run_eval.sbatch   (GR_RUNS / GR_EVAL / GR_LOCI override dirs)
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
PYE=/home/jwestrob/miniconda3/envs/elsa/bin/python
P=$RD/ws_dpann/package
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue TMPDIR=$RD/tmp NUMBA_CACHE_DIR=$RD/tmp/numba_gr_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
cd $RD/grammar/code
LO=${GR_LOCI:-$RD/grammar/loci}
ENG=$RD/engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3
for step in ${STEPS:-e123 loci score learned}; do
  echo "=== step $step $(date)"
  case $step in
    e123|loci|learned|dc) "$PY" -u evaluate.py $step "$T" ;;
    score)
      for d in "$LO"/*/; do
        d=${d%/}
        [[ -f "$d/loci_all_genomes.parquet" ]] || continue
        echo "--- scoring $d"
        "$PYE" -u "$P/scorer/score_dpann_loci.py" --pkg "$P" --loci "$d/loci_all_genomes.parquet" --out "$d/score_min2" \
            --threads "$T" --min-members 2 > "$d/score.log" 2>&1
        "$PYE" -u "$RD/engine/code/dpann_compare.py" "$d" "$ENG" > "$d/compare_vs_engine.txt"
        grep "summed excess" "$d/compare_vs_engine.txt" || true
        "$PYE" -u "$RD/ribosomal/code/ribosomal_check.py" check "$d" --threads "$T" > "$d/ribosomal.log" 2>&1 || echo "ribosomal check failed for $d"
        cat "$d/ribosomal_check.txt" 2>/dev/null || true
      done ;;
  esac
done
echo "=== end $(date)"
