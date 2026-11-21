#!/usr/bin/env python3
"""Structure-based residue check (rung 4, residues in place): from the US-align alignments of each half model onto
1O26 chain A (data/struct/usalign_raw.txt), the residue of the model superposed on T. maritima H53 R78 H79 R80 S88 R90
R147 R174, compared with the residue the sequence alignment (data/core_positions.tsv) assigns. Summary of TM-scores.
usage: python code/17_struct_residues.py"""
import re, json, pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
KEY = {53: "H", 78: "R", 79: "H", 80: "R", 88: "S", 90: "R", 147: "R", 174: "R"}
ref_res = []
for l in open("data/struct/pdb/1O26_A.pdb"):
    if l.startswith("ATOM") and l[12:16].strip() == "CA":
        ref_res.append(int(l[22:26]))
def model_res(fn):
    r = []
    for l in open(fn):
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            r.append(int(l[22:26]))
    return r
raw = open("data/struct/usalign_raw.txt").read().split("##### ")[1:]
C = pd.read_csv("data/core_positions.tsv", sep="\t")
rows = []
for blk in raw:
    head, body = blk.split("\n", 1)
    kind, a, b, mode = head.split("\t")
    if kind != "half_vs_ref":
        continue
    lines = body.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith('(":" denotes'))
    s1, mk, s2 = lines[i + 1], lines[i + 2], lines[i + 3]
    mres = model_res(f"data/struct/pdb/{a}")
    p1 = p2 = -1
    pid, half = a[:-6], a[-5]
    seqmap = C[(C.protein_id == pid) & (C.piece == f"lead_{half}")].set_index("refpos")
    rec = dict(model=a)
    for c1, m, c2 in zip(s1, mk, s2):
        if c1 != "-": p1 += 1
        if c2 != "-": p2 += 1
        if c2 != "-" and ref_res[p2] in KEY:
            rp = f"{KEY[ref_res[p2]]}{ref_res[p2]}"
            st = c1 if c1 != "-" else "-"
            rec[rp] = st
            if c1 != "-" and rp in seqmap.index:
                rec[rp + "_agree"] = int(mres[p1] == seqmap.loc[rp, "resnum"])
    rows.append(rec)
R = pd.DataFrame(rows)
R.to_csv("data/struct_residues.tsv", sep="\t", index=False)
cols = [f"{v}{k}" for k, v in KEY.items()]
print(R[["model"] + cols].to_string(index=False))
for h in ["_N.pdb", "_C.pdb"]:
    g = R[R.model.str.endswith(h)]
    print(h, {c: f"{int((g[c] == c[0]).sum())}/{len(g)}" for c in cols})
ag = [c for c in R.columns if c.endswith("_agree")]
print("structure-vs-sequence residue assignment agreement:", int(R[ag].sum().sum()), "/", int(R[ag].notna().sum().sum()))
U = pd.read_csv("data/struct/usalign.tsv", sep="\t")
h = U[U.kind == "half_vs_ref"].copy(); h["half"] = h.a.str[-5]
lead = ~h.a.str.startswith(("NYZ", "MBU"))
print("halves vs 1O26_A, TM by reference: lead N median", h[lead & (h.half == "N")].TM2.median(), "range", h[lead & (h.half == "N")].TM2.min(), h[lead & (h.half == "N")].TM2.max(),
      "| lead C median", h[lead & (h.half == "C")].TM2.median(), "range", h[lead & (h.half == "C")].TM2.min(), h[lead & (h.half == "C")].TM2.max())
nc = U[U.kind == "N_vs_C"]
print("N vs C within chain: TM (max of two normalizations) median", nc[["TM1", "TM2"]].max(axis=1).median(), "range", nc[["TM1", "TM2"]].max(axis=1).min(), nc[["TM1", "TM2"]].max(axis=1).max())
