#!/usr/bin/env python
"""Check B structural comparison (biotite login node, single-threaded serial loop): US-align (TM-align algorithm) per
pair, labeled protein first. TM1 = normalized by the labeled protein, TM2 = by the partner.
usage: usalign_pairs.py PAIRS.tsv FOLD_DIR OUT.tsv"""
import os, re, subprocess, sys
import pandas as pd
US = "/shared/software/bin/USalign"
pairs, fd, out = sys.argv[1], sys.argv[2], sys.argv[3]
P = pd.read_csv(pairs, sep="\t")
nm = lambda p: re.sub(r"[^A-Za-z0-9_.-]", "_", p)[:100]
rows = []
for r in P.itertuples(index=False):
    a, b = f"{fd}/{nm(r.protein_id)}.pdb", f"{fd}/{nm(r.partner)}.pdb"
    rec = dict(kind=r.kind, stratum=r.stratum, protein_id=r.protein_id, partner=r.partner, label=r.label)
    if not (os.path.exists(a) and os.path.exists(b)):
        rec["status"] = "missing_structure"
        rows.append(rec)
        continue
    res = subprocess.run([US, a, b, "-outfmt", "2"], capture_output=True, text=True)
    lines = [l for l in res.stdout.splitlines() if l.strip() and not l.startswith("#") and not l.startswith("Warning")]
    if res.returncode != 0 or not lines:
        rec["status"] = f"usalign_rc{res.returncode}"
        rows.append(rec)
        continue
    f = lines[-1].split("\t")
    rec.update(status="ok", TM1=float(f[2]), TM2=float(f[3]), RMSD=float(f[4]), IDali=float(f[7]), L1=int(f[8]), L2=int(f[9]), Lali=int(f[10]))
    rows.append(rec)
pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
print(len(rows), "pairs;", sum(r.get("status") == "ok" for r in rows), "ok")
