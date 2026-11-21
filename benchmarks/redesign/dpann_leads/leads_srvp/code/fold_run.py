#!/usr/bin/env python
"""Fold runner (local): ESMFold2 via the Biohub batch API (the only service allowed to receive SR-VP sequences) with the
brief's waiting rule. Runs Sharur scripts/fold_biohub.py OUT_DIR IN.faa [extra args] as a child process, reads the
per-structure files directly, and stops when every input has <name>.pdb or 20 minutes after the last new file.
usage: fold_run.py OUT_DIR IN.faa [fold_biohub args ...]     appends one line per call to OUT_DIR/../fold_ledger.tsv
(UTC, input file, items, models written) so the fold budget can be counted."""
import os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

out, faa, extra = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
names = [l[1:].split()[0] for l in open(faa) if l.startswith(">")]
todo = [n for n in names if not (out / f"{n}.pdb").exists()]
if not todo:
    print("nothing to fold"); sys.exit(0)
tmp = out.parent / f"_todo_{faa.stem}.faa"
seqs, cur = {}, None
for l in open(faa):
    if l.startswith(">"):
        cur = l[1:].split()[0]; seqs[cur] = []
    else:
        seqs[cur].append(l.strip())
tmp.write_text("".join(f">{n}\n{''.join(seqs[n])}\n" for n in todo))
log = open(out.parent / f"fold_{faa.stem}.log", "a")
p = subprocess.Popen([sys.executable, "/Users/jacob/Documents/Sandbox/Sharur/Sharur/scripts/fold_biohub.py", str(out), str(tmp)] + extra,
                     stdout=log, stderr=subprocess.STDOUT)
last_new, seen = time.time(), {n for n in todo if (out / f"{n}.pdb").exists()}
while True:
    time.sleep(20)
    now = {n for n in todo if (out / f"{n}.pdb").exists()}
    if len(now) > len(seen):
        last_new, seen = time.time(), now
    if len(seen) == len(todo) or p.poll() is not None or time.time() - last_new > 1200:
        break
if p.poll() is None:
    p.terminate()
missing = [n for n in todo if n not in seen]
with open(out.parent / "fold_ledger.tsv", "a") as fh:
    fh.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\t{faa.name}\t{' '.join(extra)}\t{len(todo)}\t{len(seen)}\n")
print(f"submitted {len(todo)}, models {len(seen)}, missing {len(missing)}: {' '.join(missing[:20])}")
