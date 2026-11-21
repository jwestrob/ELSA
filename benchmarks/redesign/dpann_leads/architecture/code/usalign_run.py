#!/usr/bin/env python3
"""US-align on the biotite login node, single-threaded serial loop over folds/pairs.tsv (query first, reference second).
TM1 = normalized by the query (member domain / control), TM2 = normalized by the reference. Records the query residue
range that aligns to the reference (from the alignment output).
usage: usalign_run.py PAIRS.tsv STR_DIR OUT.tsv"""
import os, subprocess, sys, csv
US = "/shared/software/bin/USalign"
pairs, sd, out = sys.argv[1:4]
rows = []
with open(pairs) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        a, b = f"{sd}/{r['query']}.pdb", f"{sd}/{r['ref']}.pdb"
        res = subprocess.run([US, a, b, "-outfmt", "2"], capture_output=True, text=True)
        lines = [l for l in res.stdout.splitlines() if l.strip() and not l.startswith("#") and not l.startswith("Warning")]
        rec = dict(r)
        if res.returncode or not lines:
            rec["status"] = f"rc{res.returncode}"; rows.append(rec); continue
        x = lines[-1].split("\t")
        rec.update(status="ok", TM1=x[2], TM2=x[3], RMSD=x[4], IDali=x[7], L1=x[8], L2=x[9], Lali=x[10])
        rows.append(rec)
keys = ["lead", "kind", "query", "ref", "status", "TM1", "TM2", "RMSD", "IDali", "L1", "L2", "Lali"]
with open(out, "w") as o:
    o.write("\t".join(keys) + "\n")
    for r in rows: o.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
print(len(rows), "pairs,", sum(r.get("status") == "ok" for r in rows), "ok")
