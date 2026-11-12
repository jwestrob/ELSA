#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J tok_recollect
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/recollect_%j.log

# Re-score TM confirmation for every TM batch on only the pairs sampled for each arm.
# tmprep is deterministic (seed 17, same arm order), so it regenerates the per-arm request
# list (tm_requested_<sampler>.parquet) and identical prefilter files; the TM-align outputs
# (aln_*.tsv) already on disk are reused. Pair-sampler batches run in parallel (disjoint arms),
# then token-sampler batches, so no two processes edit one stats.json at a time.
set -euo pipefail
WS=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
cd "$WS/code"
redo() {  # job_id
    local f; f=$( (ls "$WS/logs/quality_$1.log" "$WS/logs/tm_$1.log" 2>/dev/null || true) | head -1)
    local line; line=$(grep -m1 -o "tag=[^ ]* arms=[^ ]* tm=1.*" "$f")
    local tag arms tms
    tag=$(sed -E 's/.*tag=([^ ]*).*/\1/' <<< "$line")
    arms=$(sed -E 's/.*arms=([^ ]*).*/\1/' <<< "$line")
    tms=pair; grep -q "tm_sampler=token" <<< "$line" && tms=token
    local QW=$WS/work/quality/$tag
    ls "$QW"/aln_rr.tsv > /dev/null
    "$PY" -u quality.py tmprep --arms "$arms" --tmdir "$QW" --tm-sampler "$tms" > "$QW/recollect.log" 2>&1
    "$PY" -u quality.py tmcollect --arms "$arms" --tmdir "$QW" --tm-sampler "$tms" >> "$QW/recollect.log" 2>&1
    echo "== $1 $tag $tms"; grep -E "tm_conf" "$QW/recollect.log"
}
pids=()
for j in 76363 76364 76387 76396 76389 76400 76405 76414 76444 76449; do redo $j > "$WS/logs/recollect_part_$j.txt" 2>&1 & pids+=($!); done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
cat "$WS"/logs/recollect_part_*.txt
[ "$fail" = 0 ]
for j in 76420 76450; do redo $j; done
"$PY" -u quality.py collect
echo "$(date -Is) END"
