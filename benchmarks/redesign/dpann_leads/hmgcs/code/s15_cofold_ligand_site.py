#!/usr/bin/env python3
"""Acetyl-CoA placement in the homodimer co-folds: per ligand copy, the chain(s) it contacts, distance from the
thioester sulfur (S1P) and acetyl carbonyl carbon (C) to the catalytic Cys SG / His NE2 / Glu carboxylate, and the
residues within 4 A. Reference: CoA bound to the human HMGCS1 in 2P8U chain A (acetylated Cys129 = SCY); its 4-A
contact residues are mapped onto each model through the US-align residue alignment of the matching single-chain model
(OJT94780.1, JAJZNR010000018.1_8, CAISET010000016.1_25 vs 2P8U_A), and the overlap is reported."""
import re, numpy as np, pandas as pd
W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/hmgcs"
CAT = {"OJT94780.1": dict(E=120, C=168, H=330, N=442), "JAJZNR010000018.1_8": dict(E=146, C=194, H=332, N=450),
       "CAISET010000016.1_25": dict(E=84, C=116, H=238, N=270)}
def atoms(fn):
    out = []
    for l in open(fn):
        if l.startswith(("ATOM", "HETATM")):
            out.append(dict(rec=l[:6].strip(), name=l[12:16].strip(), res=l[17:20].strip(), ch=l[21], num=int(l[22:26]),
                            xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return out
def contacts(prot, lig, cut=4.0):
    L = np.array([a["xyz"] for a in lig]); s = set()
    for a in prot:
        if np.min(np.linalg.norm(L - a["xyz"], axis=1)) <= cut:
            s.add((a["ch"], a["num"], a["res"]))
    return s
# reference contacts in 6ESQ chain I
ref = atoms(f"{W}/refchains/2P8U_A.pdb")
coa = [a for a in ref if a["res"] == "COA"]
refprot = [a for a in ref if a["rec"] == "ATOM" or a["res"] == "SCY"]
# COA may be present in more than one copy/altloc; take all COA atoms of chain I
rc = contacts(refprot, coa)
print("2P8U_A CoA contact residues:", len(rc), sorted((n, r) for _, n, r in rc))
scy = [a for a in ref if a["res"] == "SCY"]
s_coa = [a for a in coa if a["name"] == "S1P"]
if scy and s_coa:
    print("2P8U_A SCY129 SG to CoA S1P:", round(min(np.linalg.norm(x["xyz"] - y["xyz"]) for x in scy if x["name"] == "SG" for y in s_coa), 2), "A")
def us_map(model, refchain="2P8U_A"):      # reference residue number -> model residue number, via US-align text
    t = open(f"{W}/us_raw/{model}__{refchain}.txt").read().rstrip().splitlines()
    i = [k for k, l in enumerate(t) if l.startswith('(":"')][0]
    s1, mk, s2 = t[i + 1], t[i + 2], t[i + 3]
    refnums = []; seen = set()
    for l in open(f"{W}/refchains/{refchain}.pdb"):
        if l.startswith("ATOM") and l[12:16].strip() == "CA" and (l[21], l[22:27]) not in seen:
            seen.add((l[21], l[22:27])); refnums.append(int(l[22:26]))
    mi = ri = -1; m = {}
    for x, f, y in zip(s1, mk, s2):
        if x != "-": mi += 1
        if y != "-": ri += 1
        if x != "-" and y != "-":
            m[refnums[ri]] = mi + 1
    return m
rows = []
for model, cat in CAT.items():
    A = atoms(f"{W}/folds/cofold_ACO/{model}.pdb")
    prot = [a for a in A if a["rec"] == "ATOM"]
    ligchains = sorted({a["ch"] for a in A if a["res"] == "ACO"})
    m = us_map(model)
    ref_mapped = {m[n] for _, n, _ in rc if n in m}
    for lc in ligchains:
        lig = [a for a in A if a["res"] == "ACO" and a["ch"] == lc]
        S = [a for a in lig if a["name"] == "S1P"][0]["xyz"]; Cc = [a for a in lig if a["name"] == "C"][0]["xyz"]
        cs = contacts(prot, lig)
        chains = sorted({c for c, _, _ in cs})
        main = max(chains, key=lambda c: sum(1 for x in cs if x[0] == c)) if chains else None
        def d(resnum, names, ch):
            pts = [a["xyz"] for a in prot if a["ch"] == ch and a["num"] == resnum and a["name"] in names]
            return round(min(min(np.linalg.norm(p - S), np.linalg.norm(p - Cc)) for p in pts), 1) if pts else None
        mod_main = {n for c, n, _ in cs if c == main}
        rec = dict(model=model, ligand_chain=lc, contact_chains="".join(chains), main_chain=main, n_contacts=len(cs),
                   d_CysSG=d(cat["C"], ["SG"], main), d_HisNE2=d(cat["H"], ["NE2", "ND1"], main), d_GluOE=d(cat["E"], ["OE1", "OE2"], main),
                   ref_contacts_mapped=len(ref_mapped), overlap_with_ref=len(mod_main & ref_mapped),
                   frac_ref_recovered=round(len(mod_main & ref_mapped) / max(1, len(ref_mapped)), 2),
                   plddt_lig=round(np.mean([float(l[60:66]) for l in open(f"{W}/folds/cofold_ACO/{model}.pdb") if l.startswith("HETATM") and l[21] == lc]) / 100, 1),
                   contact_residues=",".join(f"{r}{n}" for c, n, r in sorted(cs, key=lambda x: x[1]) if c == main))
        rows.append(rec)
T = pd.DataFrame(rows)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 200)
print(T.drop(columns=["contact_residues"]).to_string(index=False))
for r in T.itertuples():
    print(r.model, r.ligand_chain, r.contact_residues)
T.to_csv(f"{W}/cofold_ligand_site.tsv", sep="\t", index=False)
