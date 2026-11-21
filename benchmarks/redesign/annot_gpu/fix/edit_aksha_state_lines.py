"""Update only the factual Aksha state lines of the biotite CLAUDE.md and cluster-ops note (2026-11-21).

Usage: edit_aksha_state_lines.py CLAUDE_MD OPS_NOTE COMMIT PR_URL [--reverse]
Every replacement is an exact block; a missing block stops the script before any file is written.
--reverse restores the 2026-11-20 lines (used by ROLLBACK.sh).
"""
import os
import sys
import tempfile

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"


def blocks(commit, pr_url):
    claude = [
        ("## Aksha — production HMM-search default (updated 2026-11-20)",
         "## Aksha — production HMM-search default (updated 2026-11-21)"),
        ("""`aksha` and `/home/jwestrob/.local/bin/aksha` run Aksha 0.2.3 (tag `v0.2.3`, merge commit
`ceb48c8`) with runtime/CUDA 0.1.1 from `/groups/banfield/users/jwestrob/envs/aksha-0.2.3`; no
activation is needed""",
         f"""`aksha` and `/home/jwestrob/.local/bin/aksha` run Aksha 0.2.4.dev0 (unreleased build of branch
`fix/gpu-environment-attestation`, commit `{commit}`, {pr_url}) with runtime 0.1.1 and CUDA
0.1.2.dev0 from `/groups/banfield/users/jwestrob/envs/aksha-0.2.4.dev0`; no
activation is needed"""),
        ("""Previous state, kept for rollback: env `/groups/banfield/users/jwestrob/envs/aksha` (0.2.1 with
the `--cpu-parallel` patch) behind `~/.local/bin/aksha.prev`;""",
         f"""Previous state, kept for rollback: env `/groups/banfield/users/jwestrob/envs/aksha-0.2.3` (Aksha
0.2.3) behind `~/.local/bin/aksha.bak_20261121`. Rollback:
`bash {RD}/aksha_check/deploy_20261121/ROLLBACK.sh`.
Before that: env `/groups/banfield/users/jwestrob/envs/aksha` (0.2.1 with
the `--cpu-parallel` patch) behind `~/.local/bin/aksha.prev`;"""),
        ("""GPU support is installed and needs a suitable allocation and an explicit database manifest; keep
unpromoted experimental paths off. The six KOFAM manifests under
`/groups/banfield/projects/environmental/sr/srvp2020/Jacob/hmmer_gpu`
(`build/kofam-*/KOFAM*.manifest.json`, `results/pressed/KOFAM.manifest.json`) authenticate
`KOFAM.legacy_20261120`; make a new manifest before a GPU search of the live KOFAM.""",
         f"""GPU searches work on the H200 again (2026-11-21; the 2026-11-17 move to glibc 2.39 had disabled
them): the CUDA extension trusts its bias math in the built-in environments or after a one-time
exhaustive self-attestation per environment (record cached in `~/.cache/aksha/gpu-attestation`), and
an untrusted environment completes on GPU SSV plus the exact CPU path with one warning line. Check
with `aksha gpu-check --gpu-manifest DB=PATH ...` (`--attest` inside a GPU allocation). Manifests for
the live PFAM and KOFAM: `{RD}/annot_gpu/manifests/`;
the hmmer_gpu manifests authenticate older pressed files. Keep unpromoted experimental paths off."""),
    ]
    ops = [
        ("last_updated: 2026-11-20", "last_updated: 2026-11-21"),
        ("""`/home/jwestrob/.local/bin/aksha` runs Aksha 0.2.3 (deployed 2026-11-20) with
runtime/CUDA 0.1.1 in `/groups/banfield/users/jwestrob/envs/aksha-0.2.3`, without
activation.""",
         """`/home/jwestrob/.local/bin/aksha` runs Aksha 0.2.4.dev0 (unreleased, deployed 2026-11-21) with
runtime 0.1.1 and CUDA 0.1.2.dev0 in `/groups/banfield/users/jwestrob/envs/aksha-0.2.4.dev0`, without
activation."""),
    ]
    return claude, ops


def apply(path, pairs, reverse):
    text = open(path).read()
    for old, new in pairs:
        if reverse:
            old, new = new, old
        if text.count(old) != 1:
            raise SystemExit(f"{path}: expected exactly one copy of block starting {old[:60]!r}")
        text = text.replace(old, new, 1)
    return text


def write(path, text):
    mode = os.stat(path).st_mode & 0o7777
    descriptor, temporary = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".edit-")
    with os.fdopen(descriptor, "w") as stream:
        stream.write(text)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def main():
    claude_md, ops_note, commit, pr_url = sys.argv[1:5]
    reverse = "--reverse" in sys.argv[5:]
    claude_pairs, ops_pairs = blocks(commit, pr_url)
    new_claude = apply(claude_md, claude_pairs, reverse)
    new_ops = apply(ops_note, ops_pairs, reverse)
    write(claude_md, new_claude)
    write(ops_note, new_ops)
    print(("restored" if reverse else "updated"), claude_md, "and", ops_note)


if __name__ == "__main__":
    main()
