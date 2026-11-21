#!/usr/bin/env python3
"""Write search_<name>.sbatch (shard array) and finish_<name>.sbatch for the CPU Pfam run (run locally;
the files are copied to $A/code/cpu/ on biotite).

Shard counts: 45 min per shard on 24 CPUs at the measured 734 CPU-s per 4,000 proteins (0.18 CPU-s per protein,
0.57 CPU-ms per residue; validation job 119815), rounded so the 4-task throttle runs full waves:
testset3 8 x 319,676; srvp 8 x 401,469; lut 12 x 375,587 proteins.
"""
import re
from pathlib import Path

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
A = f"{RD}/annot_gpu"
EXCLUDE = "node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14"
# name: (output dir, source FASTA, proteins, shards, consecutive gids, append to STATUS_ALL)
RUNS = {
    "smoke1": (f"{A}/cpu_smoke/smoke1", f"{A}/validation/sample4000.faa", 4000, 1, False, 0),
    "smoke3": (f"{A}/cpu_smoke/smoke3", f"{A}/validation/sample4000.faa", 4000, 3, False, 0),
    "srvp": (f"{A}/srvp", f"{RD}/ws_srvp_bins/package/universe/proteins.faa", 3211751, 8, True, 1),
    "testset3": (f"{A}/testset3", f"{RD}/testset3/package/proteins_gid.faa", 2557405, 8, True, 1),
    "lut": (f"{A}/lut", f"{RD}/slice_lut/testset/package_k8/proteins_gid.faa", 4507042, 12, True, 1),
}

SEARCH = r"""#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J pfam_@NAME@
#SBATCH --array=0-@LAST@%4
#SBATCH --exclude=@EXCLUDE@
#SBATCH -o @O@/logs/search_%A_%a.log

# Pfam annotation of @NAME@ (@NPROT@ proteins, @K@ shards), one shard per array task, at most 4 tasks running.
# Aksha 0.2.3 in CPU mode, installed PFAM (Pfam 38.1, 27,481 models), --cut_ga, threads = $SLURM_CPUS_ON_NODE.
# Writes search/shard_NNN/ (Aksha output), search/shard_NNN.aksha.log and, once Aksha exited 0 and loaded every
# protein of the shard, search/shard_NNN.done (JSON). A failure writes search/shard_NNN.failed.
set -euo pipefail
for v in $(compgen -e | grep -E '^(ASTRA_|_ASTRA_|PLAN7_)' || true); do unset "$v"; done
export PYTHONDONTWRITEBYTECODE=1
AKSHA=/home/jwestrob/.local/bin/aksha
O=@O@
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
"""

FINISH = r"""#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J pfmerge_@NAME@
#SBATCH --exclude=@EXCLUDE@
#SBATCH -o @O@/logs/finish_%j.log

# Finishing job for @NAME@, submitted with --dependency=afterany:<search array> and ARRAY_JOB=<search array> exported.
# merge.py checks that every shard finished and that the shards cover every input protein exactly once, then merges
# the shard tables into PFAM.parquet (one process per shard, pyarrow threads over $SLURM_CPUS_ON_NODE CPUs).
# STATUS is written last: "DONE <UTC> rows <n> proteins_with_hit <n> shards <n> database PFAM only" or
# "FAILED <reason>". @ALLNOTE@
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
A=@A@
O=@O@
UPDATE_ALL=@UPDATE_ALL@

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
"""


def main():
    here = Path(__file__).resolve().parent
    for name, (o, faa, nprot, k, _consec, upd) in RUNS.items():
        allnote = ("After STATUS, under a lock on STATUS_ALL.lock, the job that finds STATUS present for all three "
                   "collections (srvp, testset3, lut) appends their three lines to $A/STATUS_ALL."
                   if upd else "Smoke run: STATUS_ALL is left untouched.")
        sub = {"@NAME@": name, "@O@": o, "@A@": A, "@NPROT@": f"{nprot:,}", "@K@": str(k), "@LAST@": str(k - 1),
               "@EXCLUDE@": EXCLUDE, "@UPDATE_ALL@": str(upd), "@ALLNOTE@": allnote}
        for tmpl, stem in ((SEARCH, "search"), (FINISH, "finish")):
            text = tmpl
            for a, b in sub.items():
                text = text.replace(a, b)
            assert not re.search(r"@[A-Z_]+@", text), f"unfilled placeholder in {stem}_{name}"
            (here / f"{stem}_{name}.sbatch").write_text(text)
            print("wrote", here / f"{stem}_{name}.sbatch")


if __name__ == "__main__":
    main()
