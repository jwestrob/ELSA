#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -J locus_search
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/locus_search_%j.log
# Locus-search benchmark. usage: STAGE=stage1|stage2|summary|posthoc|phase2|phase2scan|seqctl|seqctl3 sbatch --export=ALL run_ls.sbatch
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
LS=$RD/locus_search
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T PYTHONDONTWRITEBYTECODE=1
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue
export TMPDIR=$LS/tmp NUMBA_CACHE_DIR=$LS/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) host=$(hostname) threads=$T job=$SLURM_JOB_ID stage=${STAGE:?}"
cd "$LS/code"
"$PY" -u selftest.py
case "$STAGE" in
  phase2scan)
    # profile scan + scoring only (reuses the odd-genome engine output)
    "$PY" -u phase2.py scan --out "$LS/out" --threads "$T" ${PHASE2_ARGS:-} ;;
  seqctl)
    # sequence-rank control: MMseqs2 search of the member genes against all proteins, then S1/S2 benchmark
    MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
    D=$LS/out/seqsearch
    "$PY" -u seq_control.py query --out "$LS/out"
    "$MM" createdb "$D/query.faa" "$D/qdb"
    "$MM" createdb "$RD/ws_dpann/work/proteins_gid.faa" "$D/tdb"
    "$MM" search "$D/qdb" "$D/tdb" "$D/res" "$D/tmp" -s 7.5 -e 10 --max-seqs 20000 --threads "$T"
    "$MM" convertalis "$D/qdb" "$D/tdb" "$D/res" "$D/hits.tsv" --format-output query,target,bits,evalue --threads "$T"
    "$PY" -u seq_control.py topk --out "$LS/out" --threads "$T"
    "$PY" -u seq_control.py bench --out "$LS/out" --threads "$T" ;;
  seqctl3)
    # embeddings on top of sequence ranks: S3 = max(token, sequence rank, w * embedding rank)
    "$PY" -u seq_control.py bench3 --out "$LS/out" --threads "$T" ;;
  posthoc)
    # post-hoc analyses of stage 2 (localization, per-type AP, which targets are found)
    "$PY" -u posthoc.py --out "$LS/out" ;;
  phase2)
    # phase-0 negative control: reach fractions for different-type instance pairs
    "$PY" -u phase0_control.py --out "$LS/out" --threads "$T"
    # modules from odd genomes only: recurrence engine (v1.5, as committed) on a reindexed odd-genome universe
    "$PY" -u phase2.py prep --out "$LS/out" --threads "$T"
    OU=$LS/out/phase2/odd_universe
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/jwestrob/miniconda3/envs/elsa/bin/python -u \
        "$RD/engine/code/recurrence.py" --arm odd__mmseqs__seq__e3c50s75 --universe "$OU" --tokens "$OU/tokens.parquet" \
        --w 3 --null genome --benchmark none --all-genome-loci --out "$LS/out/phase2/engine" --threads "$T"
    "$PY" -u phase2.py scan --out "$LS/out" --threads "$T" ${PHASE2_ARGS:-} ;;
  *)      "$PY" -u bench.py "$STAGE" --out "$LS/out" --threads "$T" ${BENCH_ARGS:-} ;;
esac
echo "$(date -Is) finished"
