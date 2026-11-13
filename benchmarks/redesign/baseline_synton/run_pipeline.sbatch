#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J bs_synton
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/baseline_synton/logs/bs_synton_%j.log
# baseline_synton: the synton pipeline of Herve, Cury & Bernheim 2026 (repository commit 84baf14), cells run verbatim.
# usage: sbatch --export=ALL,DATASET=spacedust|dpann,FAMILY=faithful|matched,RUNNAME=<name> run_pipeline.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
BS=$RD/baseline_synton; C=$BS/code; ENV=$BS/env; SRC=$BS/src/bacterial-genome-syntons
T="${SLURM_CPUS_ON_NODE:?}"
export PATH="$ENV/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
case "${DATASET:?}" in
  spacedust) UNI=$RD/universe; FAA=$RD/ws_tokens/work/mmseqs/all.faa
             MATCHED=${MATCHED_TOKENS:-} ;;
  dpann)     UNI=$RD/ws_dpann/package/universe; FAA=$RD/ws_dpann/work/proteins_gid.faa
             MATCHED=$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet ;;
  *) echo "unknown DATASET"; exit 2 ;;
esac
RUN=$BS/runs/${RUNNAME:?}
FAM=$BS/fam/$DATASET
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID dataset=$DATASET family=${FAMILY:?} run=$RUN"
echo "repo HEAD $(git -C "$SRC" rev-parse HEAD)"
echo "duckdb CLI $(duckdb --version) python duckdb $(python -c 'import duckdb; print(duckdb.__version__)')"
mkdir -p "$RUN/data" "$RUN/timing" "$RUN/cell_logs" "$RUN/home"
EPY=/home/jwestrob/miniconda3/envs/elsa/bin/python     # scorer environment (read-only use)

# ---- reference gate first (scores only stored references)
if [[ "${GATE:-0}" == 1 ]]; then
  stamp "START gate"
  if [[ "$DATASET" == spacedust ]]; then
    OMP_NUM_THREADS=1 /usr/bin/time -v -o "$BS/logs/gate_${SLURM_JOB_ID}.time" "$EPY" -u "$C/score_spacedust.py" gate --threads "$T"
  else
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS="$T" NUMBA_CACHE_DIR="$BS/tmp/numba_$SLURM_JOB_ID" \
      /usr/bin/time -v -o "$BS/logs/gate_${SLURM_JOB_ID}.time" "$EPY" -u "$C/score_dpann.py" refcheck --threads "$T"
  fi
fi

# ---- families
if [[ "$FAMILY" == faithful ]]; then
  if [[ ! -s "$FAM/clu50.tsv" ]]; then
    stamp "START families (linclust cascade)"
    /usr/bin/time -v -o "$RUN/timing/families.time" bash "$C/fam_cascade.sh" "$FAA" "$FAM" "$T"
  fi
  python -u "$C/make_inputs.py" cascade --universe "$UNI" --tsv99 "$FAM/clu99.tsv" --tsv95 "$FAM/clu95.tsv" \
      --tsv80 "$FAM/clu80.tsv" --tsv50 "$FAM/clu50.tsv" --out "$RUN/data/clusters.parquet"
else
  python -u "$C/make_inputs.py" tokens --universe "$UNI" --tokens "${MATCHED:?}" --out "$RUN/data/clusters.parquet"
fi
python -u "$C/make_inputs.py" genes --universe "$UNI" --out "$RUN/data/genes.parquet"

# ---- their code, unmodified: scripts copied byte-identical, notebook cells extracted verbatim
rm -rf "$RUN/scripts" "$RUN/cells"
mkdir -p "$RUN/scripts" "$RUN/results"
cp -p "$SRC"/scripts/*.py "$RUN/scripts/"
python "$C/nb2sh.py" "$SRC/compute_syntons.ipynb" "$RUN/cells" > /dev/null
(cd "$SRC/scripts" && sha256sum *.py) > "$RUN/scripts.sha256"
(cd "$RUN/scripts" && sha256sum -c "$RUN/scripts.sha256")
# DuckDB CLI defaults are set through ~/.duckdbrc (HOME redirected to the run directory for the cells only):
# memory_limit 75% of the allocation, threads = allocation, temp files on /groups. Execution settings only.
MEMGB=$(( T * 4000 * 3 / 4 / 1000 ))
printf "SET memory_limit='%dGB';\nSET threads=%d;\nSET temp_directory='%s';\n" "$MEMGB" "$T" "$RUN/duck_tmp" > "$RUN/home/.duckdbrc"
cat "$RUN/home/.duckdbrc"
export TMPDIR="$RUN/tmp"; mkdir -p "$TMPDIR"
cd "$RUN"
HOME="$RUN/home" duckdb -c "select current_setting('memory_limit') as mem, current_setting('threads') as thr"
for f in cells/cell_*.sh; do
  i=$(basename "$f" .sh)
  if [[ -e "timing/$i.done" ]]; then stamp "skip $i (done)"; continue; fi
  stamp "START $i: $(head -1 "$f")"
  t0=$(date +%s)
  HOME="$RUN/home" /usr/bin/time -v -o "timing/$i.time" bash -e "$f" > "cell_logs/$i.log" 2>&1 || { tail -30 "cell_logs/$i.log"; exit 1; }
  touch "timing/$i.done"
  stamp "DONE $i in $(( $(date +%s) - t0 )) s; max RSS $(grep 'Maximum resident' "timing/$i.time" | awk '{print $NF}') kB"
  tail -3 "cell_logs/$i.log"
done
ls -la results
[[ -s results/failed_replicons.txt ]] && { echo "FAILED REPLICONS:"; wc -l results/failed_replicons.txt; }

# ---- loci + statistics (gold-free)
stamp "START post"
/usr/bin/time -v -o "timing/post.time" python -u "$C/synton_loci.py" --run "$RUN" --universe "$UNI"
stamp "post done"

# ---- scoring (optional; gate must have passed)
if [[ "${SCORE:-0}" == 1 ]]; then
  ARGS=(--loci "$RUN/post/loci.parquet" --label "$RUNNAME")
  [[ -s "$RUN/post/loci_paperq.parquet" ]] && ARGS+=(--loci "$RUN/post/loci_paperq.parquet" --label "${RUNNAME}_paperq")
  if [[ "$DATASET" == spacedust ]]; then
    OMP_NUM_THREADS=1 "$EPY" -u "$C/score_spacedust.py" score "${ARGS[@]}" --tag "$RUNNAME" --threads "$T"
  else
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS="$T" NUMBA_CACHE_DIR="$BS/tmp/numba_$SLURM_JOB_ID" \
      "$EPY" -u "$C/score_dpann.py" score "${ARGS[@]}" --tag "$RUNNAME" --threads "$T"
  fi
fi
stamp "ALL DONE"
