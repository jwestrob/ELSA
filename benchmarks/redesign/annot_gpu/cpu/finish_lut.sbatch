#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J pfmerge_lut
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/lut/logs/finish_%j.log

# Finishing job for lut, submitted with --dependency=afterany:<search array> and ARRAY_JOB=<search array> exported.
# merge.py checks that every shard finished and that the shards cover every input protein exactly once, then merges
# the shard tables into PFAM.parquet (one process per shard, pyarrow threads over $SLURM_CPUS_ON_NODE CPUs).
# STATUS is written last: "DONE <UTC> rows <n> proteins_with_hit <n> shards <n> database PFAM only" or
# "FAILED <reason>". After STATUS, under a lock on STATUS_ALL.lock, the job that finds STATUS present for all three collections (srvp, testset3, lut) appends their three lines to $A/STATUS_ALL.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
A=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu
O=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/lut
UPDATE_ALL=1

update_all() {
    [ "$UPDATE_ALL" = 1 ] || return 0
    local c
    for c in srvp testset3 lut; do [ -s "$A/$c/STATUS" ] || return 0; done
    for c in srvp testset3 lut; do echo "$c $(cat "$A/$c/STATUS")"; done >> "$A/STATUS_ALL"
}
write_status() {
    (
        flock 9
        echo "$1" > "$O/STATUS.tmp" && mv "$O/STATUS.tmp" "$O/STATUS"
        update_all
    ) 9> "$A/STATUS_ALL.lock"
}
fail() {
    trap - ERR TERM
    local reason="$1"
    if [ -s "$O/merge_fail_reason.txt" ]; then reason="$(head -c 800 "$O/merge_fail_reason.txt" | tr '\n' ' ')"; fi
    write_status "FAILED $reason ($(date -u +%FT%TZ), job $SLURM_JOB_ID)"
    echo "FAILED: $reason"
    exit 1
}
trap 'rc=$?; fail "exit ${rc} at line ${LINENO}: ${BASH_COMMAND}"' ERR
trap 'fail "terminated by signal"' TERM

echo "host $(hostname)  job $SLURM_JOB_ID  start $(date -u +%FT%TZ)  CPUs $SLURM_CPUS_ON_NODE  array ${ARRAY_JOB:-unset}"
if [ -n "${ARRAY_JOB:-}" ]; then
    sacct -j "$ARRAY_JOB" -X -n -P -o JobID,State,ExitCode,Elapsed,NodeList > "$O/logs/search_sacct.txt" || true
    cat "$O/logs/search_sacct.txt" || true
fi
if [ -e "$O/STATUS" ]; then mv "$O/STATUS" "$O/STATUS.prev_$(date -u +%Y%m%dT%H%M%SZ)"; fi
rm -f "$O/merge_fail_reason.txt"
failed=$(find "$O/search" -maxdepth 1 -name 'shard_*.failed' | wc -l)
echo "shards with a .failed marker: $failed"
$PY -I "$A/code/cpu/merge.py" --outdir "$O" --meta "$A/meta/pfam_meta.tsv" --threads "$SLURM_CPUS_ON_NODE"
fields=$($PY -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["status_fields"])' "$O/PFAM_summary.json")
trap - ERR TERM
write_status "DONE $(date -u +%FT%TZ) $fields"
cat "$O/STATUS"
echo "FINISH_DONE $(date -u +%FT%TZ)"
