#!/bin/bash
# Roll back the Aksha 0.2.4.dev0 promotion of 2026-11-21: launcher, the factual Aksha lines of CLAUDE.md and of
# the cluster-ops note, and the GPU attestation cache it may have created. Each step acts only on what the
# promotion left behind, so it is safe after a partial run. Databases, registry and manifests were never touched;
# /groups/banfield/users/jwestrob/envs/aksha-0.2.3 was never modified.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
D=$RD/aksha_check/deploy_20261121
L=/home/jwestrob/.local/bin/aksha
TAG=20261121
CLAUDE=/home/jwestrob/.claude/CLAUDE.md
OPS=/home/jwestrob/jwestrob/obsidian-vault/biotite-cluster-ops.md
CACHE=/home/jwestrob/.cache/aksha/gpu-attestation
PY=/groups/banfield/users/jwestrob/envs/aksha-0.2.3/bin/python
NOW=$(date -u +%Y%m%dT%H%M%SZ)
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }

if [ -e "$L.bak_$TAG" ]; then
    cp -p "$L" "$L.0.2.4.dev0.$NOW"
    mv -f "$L.bak_$TAG" "$L"
    say "launcher restored from $L.bak_$TAG (0.2.4.dev0 launcher kept as $L.0.2.4.dev0.$NOW)"
fi
if [ -s "$D/edit_args" ]; then
    COMMIT=$(sed -n 1p "$D/edit_args"); PR_URL=$(sed -n 2p "$D/edit_args")
    if grep -q "Aksha 0.2.4.dev0" "$CLAUDE" "$OPS" 2>/dev/null; then
        if env -u PYTHONPATH -u PYTHONHOME "$PY" -I "$D/edit_aksha_state_lines.py" "$CLAUDE" "$OPS" "$COMMIT" "$PR_URL" --reverse; then
            say "CLAUDE.md and ops-note Aksha lines restored"
        else
            say "WARNING: the edited blocks changed since the promotion; restore by hand from $D/logs/CLAUDE.md.before and $D/logs/ops.before"
        fi
    fi
fi
if [ "$(cat "$D/cache_state_before" 2>/dev/null)" = absent ] && [ -e "$CACHE" ]; then
    mv "$CACHE" "$CACHE.rolledback_$NOW"
    say "GPU attestation cache moved to $CACHE.rolledback_$NOW (0.2.3 never reads it)"
fi
say "rollback finished"
