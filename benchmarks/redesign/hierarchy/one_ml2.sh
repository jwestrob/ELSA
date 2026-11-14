#!/bin/bash
# One engine arm (recurrence_ml.py; one level = v1.5), the package scorer, the paired comparison against the base
# and the standing ribosomal-locus check.
# usage: one_ml2.sh <arm> <levels spec name=path,...> <threads>
set -euo pipefail
ARM=$1; LEVELS=$2; TH=$3
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
OUT=$RD/engine/results_dpann_hier
BASE=$RD/engine/results_dpann_v2/dpann__mmseqs__seq__e3c50s75__w3
LOG=$RD/hierarchy/logs/${ARM}.log
mkdir -p $RD/hierarchy/logs
export NUMBA_NUM_THREADS=$TH NUMBA_CACHE_DIR=$RD/tmp/numba_${SLURM_JOB_ID:-x}_${ARM}
mkdir -p "$NUMBA_CACHE_DIR"
O=$OUT/${ARM}__w3
"$PY" -u $RD/hierarchy/code/recurrence_ml.py --arm "$ARM" --levels "$LEVELS" --universe $P/universe --w 3 \
    --out $OUT --threads $TH > $LOG 2>&1
"$PY" -u $P/scorer/score_dpann_loci.py --pkg $P --loci $O/loci_all_genomes.parquet --out $O/score_min2 \
    --threads $TH --min-members 2 > $O/score_min2.log 2>&1
"$PY" -u $RD/engine/code/dpann_compare.py $O $BASE > $O/compare_vs_base.txt
grep "summed" $O/compare_vs_base.txt | sed "s/^/$ARM /"
"$PY" -u $RD/ribosomal/code/ribosomal_check.py check $O --threads $TH 2>> $LOG | sed "s/^/$ARM /" || echo "$ARM RIBOSOMAL check failed"
rm -rf "$NUMBA_CACHE_DIR"
