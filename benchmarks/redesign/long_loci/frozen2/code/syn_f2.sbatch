#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf2_syn
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen2/logs/syn_%j.log
# long_loci frozen run 2 (SR-VP bins), J6 / J7: the synton pipeline of Herve, Cury & Bernheim 2026 (commit 84baf14, cells run
# verbatim) on the lut test set. baseline_synton/code/run_pipeline.sbatch copied: DATASET = lut (package universe and
# FASTA), outputs under $FZ/synton; reference gate and scoring removed (scored by fscore.py). baseline_synton env and
# repository used read-only.
# usage: sbatch --export=ALL,FAMILY=faithful|matched,RUNNAME=<name> syn_f.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
FZ=$RD/long_loci/frozen2; C=$FZ/code
BS=$RD/baseline_synton; ENV=$BS/env; SRC=$BS/src/bacterial-genome-syntons
PKG=$RD/ws_srvp_bins/package
T="${SLURM_CPUS_ON_NODE:?}"
export PATH="$ENV/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
UNI=$PKG/universe; FAA=$PKG/universe/proteins.faa
MATCHED=$FZ/tok/mmseqs__seq__e3c50s75_rescue3__srvp.parquet
RUN=$FZ/synton/runs/${RUNNAME:?}
FAM=$FZ/synton/fam/lut
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID dataset=srvp family=${FAMILY:?} run=$RUN"
echo "repo HEAD $(git -C "$SRC" rev-parse HEAD)"
echo "duckdb CLI $(duckdb --version) python duckdb $(python -c 'import duckdb; print(duckdb.__version__)')"
mkdir -p "$RUN/data" "$RUN/timing" "$RUN/cell_logs" "$RUN/home"
T0=$(date +%s)

# ---- families
if [[ "$FAMILY" == faithful ]]; then
  if [[ ! -s "$FAM/clu50.tsv" ]]; then
    stamp "START families (linclust cascade)"
    /usr/bin/time -v -o "$RUN/timing/families.time" bash "$C/fam_cascade.sh" "$FAA" "$FAM" "$T"
  fi
  python -u "$C/make_inputs.py" cascade --universe "$UNI" --tsv99 "$FAM/clu99.tsv" --tsv95 "$FAM/clu95.tsv" \
      --tsv80 "$FAM/clu80.tsv" --tsv50 "$FAM/clu50.tsv" --out "$RUN/data/clusters.parquet"
else
  python -u "$C/make_inputs.py" tokens --universe "$UNI" --tokens "$MATCHED" --out "$RUN/data/clusters.parquet"
fi
python -u "$C/make_inputs.py" genes --universe "$UNI" --out "$RUN/data/genes.parquet"

# ---- their code, unmodified: scripts copied byte-identical, notebook cells extracted verbatim
rm -rf "$RUN/scripts" "$RUN/cells"
mkdir -p "$RUN/scripts" "$RUN/results"
cp -p "$SRC"/scripts/*.py "$RUN/scripts/"
python "$C/nb2sh.py" "$SRC/compute_syntons.ipynb" "$RUN/cells" > /dev/null
(cd "$SRC/scripts" && sha256sum *.py) > "$RUN/scripts.sha256"
(cd "$RUN/scripts" && sha256sum -c "$RUN/scripts.sha256")
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
stamp "post done; total $(( $(date +%s) - T0 )) s"
stamp "ALL DONE"
