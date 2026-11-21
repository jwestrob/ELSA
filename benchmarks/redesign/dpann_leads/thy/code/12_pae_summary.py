#!/usr/bin/env python3
"""Summarize the PAE matrices of the dimer + ligand models by domain block (N and C halves of chains A and B, cut from
data/struct/cuts.json), save data/fold/dimer_lig_pae_blocks.tsv, then rewrite each JSON without the full PAE matrix
(keeps the per-workstream disk budget; pTM, ipTM, pLDDT and pair-chain ipTM stay).
usage: python code/12_pae_summary.py"""
import json, glob, os, numpy as np, pandas as pd
cuts = json.load(open("data/struct/cuts.json"))
rows = []
for fn in sorted(glob.glob("data/fold/dimer_lig/*.json")):
    pid = os.path.basename(fn)[:-5]
    j = json.load(open(fn))
    pae = j.get("pae")
    if pae is None:
        continue
    P = np.array(pae, dtype=float)
    L = cuts.get(pid, {}).get("length")
    if L is None:
        L = (P.shape[0] - 0) // 2  # negative control: whole chains only
    k = cuts.get(pid, {}).get("struct_cut", L // 2)
    blocks = {"A_N": range(0, k), "A_C": range(k, L), "B_N": range(L, L + k), "B_C": range(L + k, 2 * L)}
    rec = dict(model=pid, L=L, cut=k, n_tokens=P.shape[0], ptm=j.get("ptm"), iptm=j.get("interface_ptm"))
    names = list(blocks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia, ib = np.array(blocks[a]), np.array(blocks[b])
            rec[f"{a}~{b}"] = round(float((P[np.ix_(ia, ib)].mean() + P[np.ix_(ib, ia)].mean()) / 2), 1)
    rows.append(rec)
    j.pop("pae", None)
    json.dump(j, open(fn, "w"))
R = pd.DataFrame(rows)
R.to_csv("data/fold/dimer_lig_pae_blocks.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(R.to_string(index=False))
