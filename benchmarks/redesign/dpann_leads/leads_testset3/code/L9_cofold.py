#!/usr/bin/env python
"""Leads pipeline criterion A (local): geometry of substrate / cofactor co-folds (ESMFold2 all-atom through Biohub), same
read-out for candidates and the family's positive control (generalized from dpann_leads/invisible/code/s09_cofold.py).
usage: L9_cofold.py COLLECTION PLAN.tsv
PLAN.tsv columns: func, protein_id, role, model (path to the co-fold PDB), ligands (comma list of CCD codes in the model).
Per protein chain: for each curated catalytic residue (model position from data/residues_full.tsv, own-reference pair) the
minimum distance from its side-chain donor/acceptor atoms to (a) a phosphorus of a ligand, (b) any non-metal ligand atom,
(c) a metal ion; for each curated metal-binding residue (UniProt Binding site whose ligand is a metal ion) whether a
side-chain atom lies within 3.0 A of a metal. Writes data/cofold_geometry.tsv."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

C, PLAN = sys.argv[1], sys.argv[2]
H = Path(json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]["local_out"].replace(
    "R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
RES = pd.read_csv(H / "data/residues_full.tsv", sep="\t", keep_default_na=False)
SC = {"H": ("ND1", "NE2"), "C": ("SG",), "E": ("OE1", "OE2"), "D": ("OD1", "OD2"), "K": ("NZ",), "R": ("NH1", "NH2", "NE"),
      "S": ("OG",), "T": ("OG1",), "Y": ("OH",), "N": ("OD1", "ND2"), "Q": ("OE1", "NE2"), "M": ("SD",), "W": ("NE1",), "G": ("CA",)}
METALS = {"ZN", "MG", "MN", "FE", "FE2", "CO", "NI", "CU", "CA", "K", "NA"}
METAL_WORDS = ("Zn", "Mg", "Mn", "Fe", "Co", "Ni", "Cu", "Ca2", "Ca(", "K(+)", "Na(+)")


def atoms(p):
    A = []
    for l in open(p):
        if l.startswith(("ATOM", "HETATM")):
            A.append(dict(rec=l[:6].strip(), name=l[12:16].strip(), res=l[17:20].strip(), chain=l[21], resi=int(l[22:26]),
                          xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]), b=float(l[60:66])))
    return A


plan = pd.read_csv(PLAN, sep="\t", dtype={"protein_id": str})
rows = []
for p in plan.itertuples():
    mp = Path(p.model)
    if not mp.exists():
        rows.append(dict(func=p.func, protein_id=p.protein_id, role=p.role, status="no_model")); continue
    A = atoms(mp)
    js = mp.with_suffix(".json")
    j = json.load(open(js)) if js.exists() else {}
    ligs = set(str(p.ligands).split(","))
    lig = [x for x in A if x["rec"] == "HETATM" and x["res"] in ligs and x["res"] not in METALS]
    met = [x for x in A if x["rec"] == "HETATM" and x["res"] in METALS]
    P_ = [x for x in lig if x["name"].startswith("P")]
    r = RES[(RES.protein_id.astype(str) == p.protein_id) & (RES.pair == "own") & (RES.ref_func == p.func)]
    cat = r[r.feature == "Active site"]
    mlig = r[(r.feature == "Binding site") & r.ligand.apply(lambda s: any(w in s for w in METAL_WORDS))]
    out = []
    for ch in sorted({x["chain"] for x in A if x["rec"] == "ATOM"}):
        byres = {}
        for x in A:
            if x["chain"] == ch and x["rec"] == "ATOM":
                byres.setdefault(x["resi"], []).append(x)
        s = []
        for x in cat.itertuples():
            if x.model_pos == "":
                s.append(f"{x.ref_residue}{x.uniprot_pos}:unaligned"); continue
            sc = [a["xyz"] for a in byres.get(int(x.model_pos), []) if a["name"] in SC.get(x.model_residue, ())]
            parts = []
            for lab, grp in (("P", P_), ("lig", lig), ("metal", met)):
                if grp and sc:
                    parts.append(f"{lab} {min(np.linalg.norm(u - v['xyz']) for u in sc for v in grp):.1f}")
            s.append(f"{x.model_residue}{x.model_pos}[{', '.join(parts)}]")
        nz, zt = 0, []
        for x in mlig.itertuples():
            if x.model_pos == "" or x.model_residue != x.ref_residue:
                zt.append(f"{x.ref_residue}{x.uniprot_pos}->{x.model_residue or '-'}"); continue
            sc = [a["xyz"] for a in byres.get(int(x.model_pos), []) if a["name"] in SC.get(x.model_residue, ())]
            d = min((np.linalg.norm(u - m["xyz"]) for u in sc for m in met), default=99)
            nz += d <= 3.0
            zt.append(f"{x.model_residue}{x.model_pos}:{d:.1f}")
        out.append(f"chain {ch}: {'; '.join(s)} | metal ligands <= 3.0 A {nz}/{len(mlig)} ({' '.join(zt)})")
    rows.append(dict(func=p.func, protein_id=p.protein_id, role=p.role, status="ok", ptm=round(j.get("ptm") or 0, 3),
                     iptm=round(j.get("interface_ptm") or 0, 3),
                     ligand_plddt=round(float(np.mean([x["b"] for x in lig + met])), 1) if lig + met else "", readout=" || ".join(out)))
T = pd.DataFrame(rows)
T.to_csv(H / "data/cofold_geometry.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 230)
print(T.to_string(index=False))
