#!/usr/bin/env python3
"""Architecture by coordinate for OJT94780.1 (cluster A rep) and JAJZNR010000018.1_8 (cluster C rep): segments that
align to gaps in the archaeal HMGCS structure (6ESQ_I) and in the human HMGCS1 structure (2P8U_A) in the US-align
residue alignment; per segment, mean pLDDT (single-chain model), fraction of residues within 8 A of the acetyl-CoA or of
the partner chain in the homodimer co-fold, and cysteine count (zinc-ribbon check)."""
import numpy as np, pandas as pd
W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/hmgcs"
def gaps(model, ref):
    t = open(f"{W}/us_raw/{model}__{ref}.txt").read().rstrip().splitlines()
    i = [k for k, l in enumerate(t) if l.startswith('(":"')][0]
    s1, s2 = t[i + 1], t[i + 3]
    segs, st, mi = [], None, 0
    for x, y in zip(s1, s2):
        if x != "-":
            mi += 1
            if y == "-":
                st = st or mi; en = mi
            elif st:
                segs.append((st, en)); st = None
    if st: segs.append((st, en))
    return [(a, b) for a, b in segs if b - a + 1 >= 8]
def load(fn):
    A = []
    for l in open(fn):
        if l.startswith(("ATOM", "HETATM")):
            A.append((l[:6].strip(), l[21], int(l[22:26]), l[17:20], np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]), float(l[60:66])))
    return A
rows = []
for model in ["OJT94780.1", "JAJZNR010000018.1_8"]:
    seq = "".join(l.strip() for l in open(f"{W}/fold_in/single.faa").read().split(f">{model}\n")[1].split(">")[0].splitlines())
    S = load(f"{W}/folds/single/{model}.pdb")
    plddt = {}
    for rec, ch, n, rn, xyz, b in S:
        plddt.setdefault(n, b)
    C = load(f"{W}/folds/cofold_ACO/{model}.pdb")
    lig = np.array([a[4] for a in C if a[3] == "ACO"])
    chB = np.array([a[4] for a in C if a[0] == "ATOM" and a[1] == "B"])
    near_lig, near_B = set(), set()
    for rec, ch, n, rn, xyz, b in C:
        if rec == "ATOM" and ch == "A":
            if np.min(np.linalg.norm(lig - xyz, axis=1)) <= 8: near_lig.add(n)
            if np.min(np.linalg.norm(chB - xyz, axis=1)) <= 8: near_B.add(n)
    for ref in ["6ESQ_I", "2P8U_A"]:
        for a, b in gaps(model, ref):
            rr = range(a, b + 1)
            rows.append(dict(model=model, vs=ref, start=a, end=b, length=b - a + 1, pLDDT=round(np.mean([plddt[k] for k in rr]), 1),
                             frac_near_ligand=round(np.mean([k in near_lig for k in rr]), 2), frac_near_partner_chain=round(np.mean([k in near_B for k in rr]), 2),
                             cys=seq[a - 1:b].count("C"), seq=seq[a - 1:b]))
    print(model, "whole-chain mean pLDDT", round(np.mean(list(plddt.values())), 1), "| residues near partner chain:", len(near_B))
T = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 90)
print(T.to_string(index=False))
T.to_csv(f"{W}/insertion_segments.tsv", sep="\t", index=False)
