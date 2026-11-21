"""Submit the screen-2 fold items (cognate, non-cognate swaps, uncalled windows) of every block that has cleared
triage (work/s2/cleared_ranks.txt) and has not been submitted yet (any work/s2/folds_s2*/state.json).
Each submission gets its own output directory work/s2/folds_s2_bNN (a running collector rewrites its own
state.json, so batches never share one). Prints the directory for the collector."""
import glob
import json
import subprocess
import sys
from pathlib import Path

W2 = Path(__file__).resolve().parent / "work" / "s2"
ranks = {int(x) for x in open(W2 / "cleared_ranks.txt").read().split()}
if len(sys.argv) > 2 and sys.argv[1] == "--ranks":
    ranks = {int(x) for x in sys.argv[2].split(",")}
have = set()
for p in glob.glob(str(W2 / "folds_s2*" / "state.json")):
    have |= set(json.load(open(p))["tasks"])
items = []
for line in open(W2 / "jobs_s2.jsonl"):
    j = json.loads(line)
    fid = j["fold_id"]
    if j["cls"] == "positive" or fid in have:
        continue
    if int(fid.split("_")[1]) in ranks:
        items.append(line)
print(len(items), "items to submit;", len(have), "already submitted")
if items and "--go" in sys.argv:
    n = len(glob.glob(str(W2 / "folds_s2_b*"))) + 2
    out = W2 / f"folds_s2_b{n:02d}"
    out.mkdir()
    jf = out / "batch_jobs.jsonl"
    jf.write_text("".join(items))
    subprocess.run([sys.executable, str(W2.parent.parent / "biohub_cofold.py"), "submit", "--jobs", str(jf),
                    "--out", str(out)], check=True)
    print("OUTDIR", out)
