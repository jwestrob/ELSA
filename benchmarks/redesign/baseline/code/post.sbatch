#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J bl_post
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/baseline_post_%j.log

# Calibrate legacy blocks; write locus tables (legacy: all block intervals and their union as a single
# reading, as shipped and calibrated; engine: conserved@2 and boundary rule H as single readings); score
# every table exactly as the engine arms are scored (score_dpann_loci.py --min-members 2, and
# --max-span 30); ribosomal check; boundary precision; comparison tables against the engine readings
# (engine full table = the 11-reading table scored until now, existing scores reused).
# usage: ARMS="raw zca" [SKIP_PREPARE=1] [TAG=...] sbatch --export=ALL post.sbatch
set -uo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
C=$RD/baseline/code
R=$RD/baseline/results
T="${SLURM_CPUS_ON_NODE:?}"
BASE=$RD/engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3
RESC=$RD/engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3
ARMS="${ARMS:-raw zca}"
TAG="${TAG:-full}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="$T" NUMBA_THREADING_LAYER=workqueue
export TMPDIR="$RD/baseline/tmp" NUMBA_CACHE_DIR="$RD/baseline/tmp/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "job ${SLURM_JOB_ID} arms '$ARMS' threads $T host $(hostname) $(date)"
if [[ -z "${SKIP_PREPARE:-}" ]]; then
    /usr/bin/time -v "$PY" -u "$C/post.py" prepare --arms $ARMS || { echo "prepare FAILED"; exit 1; }
fi
LEG=""
for arm in $ARMS; do LEG="$LEG $arm ${arm}_cal ${arm}_union ${arm}_cal_union"; done
ENG="engine_base_c2 engine_rescue3_c2 engine_base_H engine_rescue3_H"
for v in $ENG $LEG; do
    O=$R/$v
    # scorer workers allowed by memory: ~130 B per (draw x interval) element on the largest gold contig
    W=$("$PY" -c "import json; s=json.load(open('$O/stats.json')); L=max(1, s['distinct_per_gold_contig_max']); print(max(1, min($T // 2, int(110e9 / (201 * L * 130)))))")
    echo "== $v  scorer workers per pass: $W  $(date)"
    if [[ ! -f "$O/score_min2/instance_scores.parquet" ]]; then
        /usr/bin/time -v "$PY" -u "$P/scorer/score_dpann_loci.py" --pkg "$P" --loci "$O/loci.parquet" --out "$O/score_min2" \
            --threads "$W" --min-members 2 > "$O/score_min2.log" 2>&1 &
        p1=$!
        /usr/bin/time -v "$PY" -u "$P/scorer/score_dpann_loci.py" --pkg "$P" --loci "$O/loci.parquet" --out "$O/score_min2_span30" \
            --threads "$W" --min-members 2 --max-span 30 > "$O/score_min2_span30.log" 2>&1 &
        p2=$!
        wait $p1 || echo "scorer min2 FAILED for $v"
        wait $p2 || echo "scorer span30 FAILED for $v"
        grep -E "Elapsed|Maximum resident" "$O/score_min2.log" || true
    fi
    "$PY" -u "$C/ribo_blocks.py" --loci "$O/loci.parquet" --name "$v" --out "$O" --threads "$T" || echo "ribosomal FAILED for $v"
done
# engine full tables (11 readings) and conserved@2 through the same ribosomal input side
for ref in base rescue3; do
    D=$BASE; [[ $ref == rescue3 ]] && D=$RESC
    "$PY" -u "$C/ribo_blocks.py" --loci "$D/loci_all_genomes.parquet" --name "engine_${ref}_full" --out "$R/engine_${ref}_full" --threads "$T" || true
done
SETS="engine_base_full=$BASE/loci_all_genomes.parquet:$BASE/score_min2 engine_rescue3_full=$RESC/loci_all_genomes.parquet:$RESC/score_min2"
TSETS="engine_base_full=$BASE engine_rescue3_full=$RESC"
for v in $ENG $LEG; do SETS="$SETS $v=$R/$v/loci.parquet:$R/$v/score_min2"; TSETS="$TSETS $v=$R/$v"; done
"$PY" -u "$C/bprec.py" --out "$R/bprec_$TAG" $SETS || echo "bprec FAILED"
"$PY" -u "$C/tables.py" --out "$R/tables_$TAG" --refs engine_base_full engine_rescue3_full $ENG --sets $TSETS || echo "tables FAILED"
echo "done $(date)"
