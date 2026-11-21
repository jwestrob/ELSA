#!/usr/bin/env python3
"""Step 21: ligand placement in the ESMFold2 all-atom co-folds of L1 members with CTP, ATP and 2 Mg. For each ligand
the protein residues within 4.5 A, and the distance from the ligand to the conserved active-site residues found in
steps 17/18 (NTase: L-A-A-G motif and the LicC D107-equivalent; kinase: the APH H-G-D...N catalytic loop and the
LicA D194-equivalent Mg ligand). Writes data/cofold_L1_contacts.tsv."""
import numpy as np, pandas as pd, os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SITES = {"HII14378.1": dict(NTase=[9, 10, 11, 12, 110], kinase=[389, 391, 396, 414]),
         "MBX4196761.1": dict(NTase=[9, 10, 11, 12, 110], kinase=[388, 390, 395, 413])}
rows = []
for pid, site in SITES.items():
    prot, lig = {}, {}
    for l in open(f"folds/cofold_L1/{pid}.pdb"):
        if not l.startswith(("ATOM", "HETATM")): continue
        xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        if l[21] == "A" and l.startswith("ATOM"): prot.setdefault((int(l[22:26]), l[17:20]), []).append(xyz)
        else: lig.setdefault((l[21], l[17:20].strip()), []).append(xyz)
    js = json.load(open(f"folds/cofold_L1/{pid}.json"))
    for (ch, nm), X in lig.items():
        X = np.array(X)
        d = {k: np.linalg.norm(np.array(v)[:, None, :] - X[None], axis=2).min() for k, v in prot.items()}
        near = sorted([k for k, v in d.items() if v <= 4.5])
        dn = {s: round(min(d[k] for k in d if k[0] in res), 1) for s, res in site.items()}
        rows.append(dict(protein=pid, ligand=nm, chain=ch, n_contacts=len(near), contacts=",".join(f"{r[1]}{r[0]}" for r in near),
                         min_dist_NTase_site=dn["NTase"], min_dist_kinase_site=dn["kinase"], ptm=js.get("ptm"), iptm=js.get("iptm")))
R = pd.DataFrame(rows); R.to_csv("data/cofold_L1_contacts.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(R.to_string(index=False))
