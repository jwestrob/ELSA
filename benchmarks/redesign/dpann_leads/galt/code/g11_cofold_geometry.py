#!/usr/bin/env python
"""galt step 11: geometry of the ESMFold2 all-atom co-folds (folds/cofold_<TAG>_<L>/<name>.pdb).
Per protein chain (A, B):
  his      nucleophilic His = middle His of the HIT motif H-x-H-x-[QSNH] (E. coli GalT H166, ADPGP H186, Apa2 H161,
           GDPGP1 H218, DPANN leads H195/H196, checked against the curated active-site records).
  nuc      nucleotide ligand copy (UPG / ADQ / B4P / GDG) whose alpha-phosphorus lies closest to that His.
  alphaP   phosphorus of the nucleotide closest to the nucleobase nitrogen centroid (B4P: PA or PD, the two
           adenosine-proximal phosphates). d_his_aP = min distance His NE2/ND1 -> alphaP.
  d_his_acc  min distance His NE2/ND1 -> acceptor phosphorus (GL1 or PO4).
  lig_plddt  mean pLDDT of the nucleotide copy atoms; iptm_pl = pair ipTM protein chain x that ligand chain.
  contact  share of curated binding residues of the control's own record (cognate ligand, UniProt Binding site
           features) with any heavy atom within 4.5 A of the nucleotide copy (controls only).
productive = d_his_aP <= 4.0 A.
Writes data/cofold_geometry.tsv."""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
F = pd.read_csv(H / "data/ref_features.tsv", sep="\t", keep_default_na=False)
CURATED = {"P09148": "UDP-alpha-D-gl", "Q9FK51": "ADP-alpha-D-gl", "P22108": "substrate"}
NUC = {"UPG", "ADQ", "B4P", "GDG"}
ACC = {"GL1", "PO4"}


def binding_positions(acc):
    if acc not in CURATED:
        return []
    x = F[(F.accession == acc) & (F.type == "Binding site") & F.ligand.str.startswith(CURATED[acc][:8])]
    return sorted({p for r in x.itertuples() for p in range(int(r.start), int(r.end) + 1)})


def read(p):
    A = []
    for l in open(p):
        if l.startswith(("ATOM", "HETATM")):
            A.append((l[:6].strip(), l[12:16].strip(), l[17:20].strip(), l[21], int(l[22:26]),
                      float(l[30:38]), float(l[38:46]), float(l[46:54]), float(l[60:66]), l[76:78].strip()))
    return pd.DataFrame(A, columns=["rec", "name", "res", "chain", "resi", "x", "y", "z", "b", "el"])


rows = []
for d in sorted((H / "folds").glob("cofold_*_L*")):
    L = d.name.split("_", 2)[2]
    tag = d.name.split("_")[1]
    for p in sorted(d.glob("*.pdb")):
        name = p.stem
        js = json.load(open(p.with_suffix(".json")))
        A = read(p)
        prot = A[A.rec == "ATOM"]
        chains = sorted(prot.chain.unique())
        lig = A[A.rec == "HETATM"]
        lchains = sorted(lig.chain.unique())
        allchains = chains + lchains
        pc = js.get("pair_chains_iptm")
        seq = "".join({"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
                       "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                       "TYR": "Y", "VAL": "V"}.get(r, "X") for r in prot[prot.chain == "A"].drop_duplicates("resi").res)
        m = [x.start() + 3 for x in re.finditer(r"(?=H.H.[QSNH])", seq)]
        his = m[0] if m else None
        acc = name.split("_")[-1] if name.startswith("ctl_") else ""
        bpos = binding_positions(acc)
        for ch in chains:
            pr = prot[prot.chain == ch]
            hx = pr[(pr.resi == his) & pr.name.isin(["NE2", "ND1"])][["x", "y", "z"]].to_numpy() if his else np.zeros((0, 3))
            best = None
            for lc in lchains:
                l = lig[lig.chain == lc]
                if l.res.iloc[0] not in NUC:
                    continue
                Ps = l[l.el == "P"]
                if l.res.iloc[0] == "B4P":
                    aP = Ps[Ps.name.isin(["PA", "PD"])]
                else:
                    N = l[(l.el == "N")][["x", "y", "z"]].to_numpy().mean(axis=0)
                    dn = np.linalg.norm(Ps[["x", "y", "z"]].to_numpy() - N, axis=1)
                    aP = Ps.iloc[[int(np.argmin(dn))]]
                dd = np.linalg.norm(hx[:, None, :] - aP[["x", "y", "z"]].to_numpy()[None, :, :], axis=2).min() if len(hx) else 99
                if best is None or dd < best[0]:
                    best = (dd, lc, l)
            if best is None:
                continue
            dd, lc, l = best
            accd = 99.0
            for ac in lchains:
                la = lig[lig.chain == ac]
                if la.res.iloc[0] in ACC and len(hx):
                    accd = min(accd, float(np.linalg.norm(hx[:, None, :] - la[la.el == "P"][["x", "y", "z"]].to_numpy()[None], axis=2).min()))
            cont = ""
            if bpos:
                heavy = pr[pr.resi.isin(bpos) & (pr.el != "H")]
                lx = l[["x", "y", "z"]].to_numpy()
                near = set()
                for r in heavy.itertuples():
                    if np.min(np.linalg.norm(lx - np.array([r.x, r.y, r.z]), axis=1)) <= 4.5:
                        near.add(r.resi)
                cont = round(len(near) / len(bpos), 2)
            ipl = ""
            if pc:
                i, j = allchains.index(ch), allchains.index(lc)
                ipl = round(pc[i][j], 3)
            rows.append(dict(tag=tag, ligset=L, protein=name, chain=ch, his=f"H{his}", nucleotide=l.res.iloc[0],
                             d_his_aP=round(float(dd), 2), d_his_acc=round(accd, 2), lig_plddt=round(l.b.mean(), 1),
                             iptm_pl=ipl, contact=cont, ptm=round(js.get("ptm") or 0, 3), iptm=round(js.get("interface_ptm") or 0, 3),
                             protein_plddt=round(pr[pr.name == "CA"].b.mean(), 1)))
T = pd.DataFrame(rows)
T["productive"] = T.d_his_aP <= 4.0
T.to_csv(H / "data/cofold_geometry.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(T.to_string(index=False))
