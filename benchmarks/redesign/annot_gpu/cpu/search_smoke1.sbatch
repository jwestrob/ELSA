#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J pfam_smoke1
#SBATCH --array=0-0%4
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/cpu_smoke/smoke1/logs/search_%A_%a.log

# Pfam annotation of smoke1 (4,000 proteins, 1 shards), one shard per array task, at most 4 tasks running.
# Aksha 0.2.3 in CPU mode, installed PFAM (Pfam 38.1, 27,481 models), --cut_ga, threads = $SLURM_CPUS_ON_NODE.
# Writes search/shard_NNN/ (Aksha output), search/shard_NNN.aksha.log and, once Aksha exited 0 and loaded every
# protein of the shard, search/shard_NNN.done (JSON). A failure writes search/shard_NNN.failed.
set -euo pipefail
for v in $(compgen -e | grep -E '^(ASTRA_|_ASTRA_|PLAN7_)' || true); do unset "$v"; done
export PYTHONDONTWRITEBYTECODE=1
AKSHA=/home/jwestrob/.local/bin/aksha
O=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/cpu_smoke/smoke1
i=$(printf '%03d' "$SLURM_ARRAY_TASK_ID")
faa=$O/shards/shard_$i.faa
out=$O/search/shard_$i
log=$O/search/shard_$i.aksha.log
done_f=$O/search/shard_$i.done
fail_f=$O/search/shard_$i.failed
trap 'rc=$?; echo "FAILED $(date -u +%FT%TZ) exit ${rc} at line ${LINENO}: ${BASH_COMMAND}" > "$fail_f"; exit $rc' ERR
trap 'echo "FAILED $(date -u +%FT%TZ) terminated by signal" > "$fail_f"; exit 143' TERM
rm -f "$fail_f"
if [ -s "$done_f" ]; then echo "shard $i already done: $(cat "$done_f")"; exit 0; fi
expected=$(awk -F'\t' -v s="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == s {print $4}' "$O/shards/shards.tsv")
n=$(grep -c '^>' "$faa")
start=$(date -u +%FT%TZ)
echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  start $start  CPUs $SLURM_CPUS_ON_NODE  shard $i  proteins $n (expected $expected)"
test -n "$expected"
test "$n" -eq "$expected"
ver=$("$AKSHA" --version)
echo "$ver"
test "$ver" = "aksha 0.2.3"
rm -rf "$out"   # only a partial result of an earlier attempt can exist here (no .done marker)
cd "$O/search"
cmd=("$AKSHA" search --installed_hmms PFAM --cut_ga --prot_in "$faa" --outdir "$out" --threads "$SLURM_CPUS_ON_NODE")
echo "command: ${cmd[*]}"
t0=$(date +%s)
/usr/bin/time -v "${cmd[@]}" 2>&1 | tee "$log"
t1=$(date +%s)
test -s "$out/PFAM_hits_df.tsv"
loaded=$(grep -oP '^Total sequences loaded: \K[0-9]+' "$log")
test "$loaded" -eq "$n"
rows=$(( $(wc -l < "$out/PFAM_hits_df.tsv") - 1 ))
user_s=$(grep -oP 'User time \(seconds\): \K[0-9.]+' "$log")
sys_s=$(grep -oP 'System time \(seconds\): \K[0-9.]+' "$log")
rss=$(grep -oP 'Maximum resident set size \(kbytes\): \K[0-9]+' "$log")
printf '{"shard": %d, "job": "%s", "host": "%s", "proteins": %d, "loaded": %d, "hit_rows": %d, "aksha": "%s", "threads": %d, "start_utc": "%s", "end_utc": "%s", "wall_s": %d, "user_s": %s, "sys_s": %s, "max_rss_kb": %s}\n' \
    "$SLURM_ARRAY_TASK_ID" "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" "$(hostname)" "$n" "$loaded" "$rows" "$ver" \
    "$SLURM_CPUS_ON_NODE" "$start" "$(date -u +%FT%TZ)" "$((t1 - t0))" "$user_s" "$sys_s" "$rss" > "$done_f.tmp"
mv "$done_f.tmp" "$done_f"
echo "SHARD_DONE $(cat "$done_f")"
