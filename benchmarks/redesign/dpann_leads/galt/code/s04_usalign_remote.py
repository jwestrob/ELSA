#!/usr/bin/env python
"""Run on the biotite login node (serial, single-threaded): US-align each model (Structure_1) onto a reference family
chain (Structure_2). Records TM1 (normalized by the model), TM2 (normalized by the reference), RMSD, Lali and the three
alignment lines (model, ':'/'.' distance marks, reference).  usage: s04_usalign_remote.py PAIRS.tsv OUT.tsv
PAIRS.tsv columns: pair_id, model, ref (paths on biotite)."""
import csv, subprocess, sys
US = "/shared/software/bin/USalign"
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
done = set()
try:
    for r in csv.DictReader(open(sys.argv[2]), delimiter="\t"):
        done.add(r["pair_id"])
except FileNotFoundError:
    pass
cols = ["pair_id", "TM1", "TM2", "RMSD", "Lali", "aln_model", "aln_marks", "aln_ref"]
new = not done
fh = open(sys.argv[2], "a")
w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
if new:
    w.writeheader()
for r in rows:
    if r["pair_id"] in done:
        continue
    out = subprocess.run(["nice", "-n", "19", US, r["model"], r["ref"]], capture_output=True, text=True).stdout.splitlines()
    rec = dict(pair_id=r["pair_id"])
    for l in out:
        if l.startswith("Aligned length="):
            f = [x.strip() for x in l.replace("Aligned length=", "").split(",")]
            rec["Lali"] = int(f[0]); rec["RMSD"] = float(f[1].split("=")[1])
        elif l.startswith("TM-score=") and "Structure_1" in l:
            rec["TM1"] = float(l.split()[1])
        elif l.startswith("TM-score=") and "Structure_2" in l:
            rec["TM2"] = float(l.split()[1])
    k = [i for i, l in enumerate(out) if l.startswith('(":" denotes')]
    if k:
        rec["aln_model"], rec["aln_marks"], rec["aln_ref"] = out[k[0] + 1], out[k[0] + 2], out[k[0] + 3]
    w.writerow(rec)
    fh.flush()
print(len(rows), "pairs")
