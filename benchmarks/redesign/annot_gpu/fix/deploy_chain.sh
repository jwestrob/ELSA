#!/bin/bash
# Unattended promotion of Aksha 0.2.4.dev0 (GPU bias-math self-attestation) on the biotite login node.
# Pattern of deploy_20261120: preconditions -> activity gate -> backup + launcher swap (auto-rollback on error)
# -> post-deploy checks -> factual Aksha lines of CLAUDE.md / cluster-ops note -> STATUS.
# Databases, the registry and every manifest are untouched; the previous environment stays in place.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
D=$RD/aksha_check/deploy_20261121
GATES=$RD/aksha_check/gpufix/gates/GATES_PASSED
NEW=/groups/banfield/users/jwestrob/envs/aksha-0.2.4.dev0
OLD=/groups/banfield/users/jwestrob/envs/aksha-0.2.3
L=/home/jwestrob/.local/bin/aksha
TAG=20261121
CLAUDE=/home/jwestrob/.claude/CLAUDE.md
OPS=/home/jwestrob/jwestrob/obsidian-vault/biotite-cluster-ops.md
CACHE=/home/jwestrob/.cache/aksha/gpu-attestation
PFAM_REFERENCE=$RD/aksha_check/out/a_pfam_cutga/PFAM_hits_df.tsv
COMMIT=${1:?branch commit}
PR_URL=${2:?pull request URL}
STATUS=$D/STATUS
SWAP=no

stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
finish() { printf '%s\n' "$*" > "$STATUS.tmp"; mv "$STATUS.tmp" "$STATUS"; stamp "STATUS: $*"; exit 0; }
fail() { finish "FAILED: $1 (swap: $SWAP; rollback command: bash $D/ROLLBACK.sh)"; }
trap 'rc=$?; [ -e "$STATUS" ] || printf "FAILED: unexpected exit rc=%s (swap: %s; rollback command: bash %s/ROLLBACK.sh)\n" "$rc" "$SWAP" "$D" > "$STATUS"' EXIT
py() { env -u PYTHONPATH -u PYTHONHOME CUDA_VISIBLE_DEVICES= "$NEW/bin/python" -I "$@"; }
mkdir -p "$D/logs"
stamp "chain pid $$ on $(hostname); commit $COMMIT; $PR_URL"

# --- 0. preconditions ---------------------------------------------------------------------------
[ -s "$GATES" ] || fail "precondition: $GATES missing"
for gate in a b c d e f; do
    grep -q "^gate $gate PASS" "$GATES" || fail "precondition: gate $gate not recorded as PASS in $GATES"
done
[ "$(py "$NEW/bin/aksha" --version)" = "aksha 0.2.4.dev0" ] || fail "precondition: new env does not report aksha 0.2.4.dev0"
grep -q "$OLD/bin/python" "$L" || fail "precondition: launcher does not point at $OLD"
[ ! -e "$L.bak_$TAG" ] || fail "precondition: $L.bak_$TAG already exists"
printf '%s\n%s\n' "$COMMIT" "$PR_URL" > "$D/edit_args"
cp -p "$CLAUDE" "$D/logs/CLAUDE.md.before" && cp -p "$OPS" "$D/logs/ops.before"
cp "$CLAUDE" "$D/logs/CLAUDE.md.dryrun" && cp "$OPS" "$D/logs/ops.dryrun"
py "$D/edit_aksha_state_lines.py" "$D/logs/CLAUDE.md.dryrun" "$D/logs/ops.dryrun" "$COMMIT" "$PR_URL" \
    > "$D/logs/edit_dryrun.log" 2>&1 || fail "precondition: CLAUDE.md / ops note blocks not found (logs/edit_dryrun.log)"
py "$D/edit_aksha_state_lines.py" "$D/logs/CLAUDE.md.dryrun" "$D/logs/ops.dryrun" "$COMMIT" "$PR_URL" --reverse \
    >> "$D/logs/edit_dryrun.log" 2>&1 || fail "precondition: reverse edit dry run failed"
cmp -s "$D/logs/CLAUDE.md.dryrun" "$D/logs/CLAUDE.md.before" && cmp -s "$D/logs/ops.dryrun" "$D/logs/ops.before" \
    || fail "precondition: forward + reverse edit dry run does not restore the files exactly"
bash -n "$D/ROLLBACK.sh" || fail "precondition: ROLLBACK.sh syntax"
if [ -e "$CACHE" ]; then echo present > "$D/cache_state_before"; else echo absent > "$D/cache_state_before"; fi
stamp "preconditions passed"

# --- 1. activity gate: no Aksha search on the login node, no running Aksha Slurm job -------------
aksha_pids() {
    ps -u jwestrob -o pid=,ppid=,comm=,args= | awk -v self="$$" '
        $1 != self && $2 != self && $3 ~ /^(python|python3|python3\.12|aksha|astra)$/ &&
        ($0 ~ /aksha search/ || $0 ~ /astra search/ || $0 ~ /aksha\.main/) { print $1 }'
}
aksha_jobs() {
    local id
    for id in $(squeue -u jwestrob -h -t RUNNING -o %i); do
        if scontrol write batch_script "$id" - 2>/dev/null | grep -q -i -E 'aksha|astra'; then echo "$id"; fi
    done
}
while :; do
    pids=$(aksha_pids)
    jobs=$(aksha_jobs)
    [ -z "$pids$jobs" ] && break
    for p in $pids; do
        stamp "waiting for Aksha process $p: $(ps -o args= -p "$p" 2>/dev/null | cut -c1-160)"
        while kill -0 "$p" 2>/dev/null; do sleep 30; done
    done
    for j in $jobs; do
        stamp "waiting for running Slurm job $j (its script uses Aksha)"
        while [ -n "$(squeue -h -j "$j" -t RUNNING -o %i 2>/dev/null)" ]; do sleep 120; done
    done
done
stamp "activity gate passed: no Aksha search running"

# --- 2. backup and swap (launcher only) ---------------------------------------------------------
do_swap() {
    cp -p "$L" "$L.bak_$TAG" || return 1
    SWAP=partial
    sed "s#$OLD/bin/#$NEW/bin/#g" "$L" > "$L.tmp" || return 1
    [ "$(grep -c "$NEW/bin/" "$L.tmp")" -ge 1 ] && ! grep -q "$OLD/bin/" "$L.tmp" || return 1
    chmod 755 "$L.tmp" && mv "$L.tmp" "$L" || return 1
    SWAP=yes
}
stamp "swapping the launcher"
if ! do_swap; then
    stamp "swap failed at state $SWAP; rolling back"
    if bash "$D/ROLLBACK.sh" >> "$D/logs/rollback.log" 2>&1; then SWAP="failed, rolled back"; else SWAP="failed, rollback failed"; fi
    fail "swap"
fi
stamp "swap done: launcher -> $NEW"
cat "$L"

# --- 3. post-deploy checks through the production launcher ---------------------------------------
v=$(env -u PYTHONPATH "$L" --version)
[ "$v" = "aksha 0.2.4.dev0" ] || fail "post-deploy: launcher reports '$v'"
env -u PYTHONPATH "$L" gpu-check --help > /dev/null || fail "post-deploy: gpu-check unavailable"
CUDA_VISIBLE_DEVICES= "$L" search --prot_in "$RD/aksha_check/input/first200.faa" --installed_hmms PFAM --cut_ga \
    --threads 4 --outdir "$D/post_pfam" > "$D/logs/post_pfam.log" 2>&1 || fail "post-deploy: PFAM smoke search"
rows=$(( $(wc -l < "$D/post_pfam/PFAM_hits_df.tsv") - 1 ))
[ "$rows" = 272 ] || fail "post-deploy: PFAM smoke gave $rows rows"
cmp -s <(sort "$D/post_pfam/PFAM_hits_df.tsv") <(sort "$PFAM_REFERENCE") || fail "post-deploy: PFAM rows differ from the 0.2.1 reference"
stamp "post-deploy: version, gpu-check, PFAM smoke 272 identical rows"

# --- 4. factual Aksha state lines ----------------------------------------------------------------
py "$D/edit_aksha_state_lines.py" "$CLAUDE" "$OPS" "$COMMIT" "$PR_URL" > "$D/logs/edit.log" 2>&1 \
    || finish "DONE with a note: launcher -> $NEW (aksha 0.2.4.dev0) and checks passed, but the CLAUDE.md / ops-note edit failed (logs/edit.log); rollback: bash $D/ROLLBACK.sh"
diff "$D/logs/CLAUDE.md.before" "$CLAUDE" > "$D/logs/CLAUDE.md.diff" || true
diff "$D/logs/ops.before" "$OPS" > "$D/logs/ops.diff" || true
finish "DONE: launcher -> $NEW (aksha 0.2.4.dev0, commit $COMMIT); previous launcher $L.bak_$TAG (env $OLD kept); PFAM smoke 272 identical rows; CLAUDE.md and ops note Aksha lines updated (logs/*.diff); rollback: bash $D/ROLLBACK.sh"
