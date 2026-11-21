#!/usr/bin/env python
"""Step 9 (criterion A): geometry of the co-folds (ESMFold2 all-atom, Biohub). Same protocol for candidates and the
positive control of each family.
  GalT-like: homodimer (--copies 2) + 2 UDP-glucose (UPG) + 2 Zn. Read-out per chain: the residue aligned to the curated
    catalytic His (human GALT H186) -> minimum distance from its ND1/NE2 to a UPG phosphorus; Zn ligands of the curated
    record (C52, C55, H115, H184) with a side-chain S/N/O atom within 3.0 A of a Zn; ligand pLDDT; ipTM.
  Amidase_3: single chain + 1 Zn. Curated Zn ligands (UniProt I6Y4D2 binding sites) with a side-chain atom within 3.0 A
    of the Zn; distance from the catalytic Glu (E200) carboxylate to the Zn.
Model positions of the curated residues come from data/residues_full.tsv (own-reference US-align pair).
Writes data/cofold_geometry.tsv."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/invisible")
RES = pd.read_csv(H / "data/residues_full.tsv", sep="\t", keep_default_na=False)
SC = {"H": ("ND1", "NE2"), "C": ("SG",), "E": ("OE1", "OE2"), "D": ("OD1", "OD2"), "K": ("NZ",), "R": ("NH1", "NH2", "NE")}


def atoms(p):
    A = []
    for l in open(p):
        if l.startswith(("ATOM", "HETATM")):
            A.append(dict(rec=l[:6].strip(), name=l[12:16].strip(), res=l[17:20].strip(), chain=l[21], resi=int(l[22:26]),
                          xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]), b=float(l[60:66])))
    return A


rows = []
for fam, d, roles in [("K00965", "galt", {"MBN2459053.1": "candidate", "MBW2990626.1": "candidate", "MEW6295551.1": "positive"}),
                      ("K01448", "amidase", {"MBI2548370.1": "candidate", "KYK24502.1": "positive"})]:
    for pid, role in roles.items():
        p = H / f"folds_cofold/{d}/{pid}.pdb"
        if not p.exists():
            rows.append(dict(family=fam, protein_id=pid, role=role, status="no_model")); continue
        A = atoms(p)
        js = json.load(open(H / f"folds_cofold/{d}/{pid}.json"))
        r = RES[(RES.protein_id == pid) & (RES.pair == "own") & (RES.ref_func == fam)]
        cat = r[r.feature == "Active site"]
        zn_l = r[(r.feature == "Binding site") & r.ligand.str.contains("Zn")]
        prot_chains = sorted({a["chain"] for a in A if a["rec"] == "ATOM"})
        zns = [a for a in A if a["res"] == "ZN"]
        lig = [a for a in A if a["res"] == "UPG"]
        P_atoms = [a for a in lig if a["name"].startswith("P")]
        rec = dict(family=fam, protein_id=pid, role=role, status="ok", ptm=round(js.get("ptm") or 0, 3),
                   iptm=round(js.get("interface_ptm") or 0, 3), ligand_plddt=round(np.mean([a["b"] for a in lig + zns]), 1) if lig + zns else "")
        out = []
        for ch in prot_chains:
            pa = [a for a in A if a["chain"] == ch and a["rec"] == "ATOM"]
            byres = {}
            for a in pa:
                byres.setdefault(a["resi"], []).append(a)
            s = []
            for x in cat.itertuples():
                if x.model_pos == "":
                    continue
                sc = [a["xyz"] for a in byres.get(int(x.model_pos), []) if a["name"] in SC.get(x.model_residue, ())]
                if P_atoms and sc:
                    dmin = min(np.linalg.norm(u - v["xyz"]) for u in sc for v in P_atoms)
                    s.append(f"{x.model_residue}{x.model_pos}-UPG_P {dmin:.1f}A")
                if zns and sc and fam == "K01448":
                    dmin = min(np.linalg.norm(u - z["xyz"]) for u in sc for z in zns)
                    s.append(f"{x.model_residue}{x.model_pos}-Zn {dmin:.1f}A")
            nz = 0; zt = []
            for x in zn_l.itertuples():
                if x.model_pos == "" or x.model_residue != x.ref_residue:
                    zt.append(f"{x.ref_residue}{x.uniprot_pos}->{x.model_residue or '-'}"); continue
                sc = [a["xyz"] for a in byres.get(int(x.model_pos), []) if a["name"] in SC.get(x.model_residue, ())]
                dmin = min((np.linalg.norm(u - z["xyz"]) for u in sc for z in zns), default=99)
                nz += dmin <= 3.0
                zt.append(f"{x.model_residue}{x.model_pos}:{dmin:.1f}")
            out.append(f"chain {ch}: {'; '.join(s)} | Zn ligands within 3.0 A {nz}/{len(zn_l)} ({' '.join(zt)})")
        rec["readout"] = " || ".join(out)
        rows.append(rec)
T = pd.DataFrame(rows)
T.to_csv(H / "data/cofold_geometry.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 220)
print(T.to_string(index=False))
