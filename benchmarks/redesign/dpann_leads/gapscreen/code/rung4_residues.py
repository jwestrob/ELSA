#!/usr/bin/env python3
"""Rung 4 residue check: map each curated UniProt residue (ref/<acc>.json features) to the experimental chain (author
numbering, offset checked against the UniProt sequence) and through the US-align structural alignment onto each model.
In place = same residue type and US-align distance marker ':' (pair within 5 A). Adds mean pLDDT (folds/summary.tsv).
Writes data/rung4_summary.tsv and data/rung4_residues.tsv."""
import json
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parents[1]
REF = {"L1_dihydroorotase": ("P05020", "1J79_A"), "L2_beta_carbonic_anhydrase": ("P61517", "1I6P_A"),
       "L3_ADP_serine_kinase": ("Q5JD03", "5X0B_A"), "L5_gamma_glutamylcysteine_ligase_GCS2": ("P77213", "1R8G_A")}
AA = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K", MET="M",
          PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V", MSE="M")
U = pd.read_csv(H / "data/rung4_usalign.tsv", sep="\t", keep_default_na=False)
pl = pd.read_csv(H / "folds/summary.tsv", sep="\t").set_index("name").protein_plddt
srow, rrow = [], []
for lead, (acc, chain) in REF.items():
    j = json.load(open(H / f"ref/{acc}.json"))
    useq = j["sequence"]["value"]
    feats = [(f["type"], f["location"]["start"]["value"], f.get("ligand", {}).get("name", "") or f.get("description", ""))
             for f in j.get("features", []) if f["type"] in ("Active site", "Binding site")]
    res = []
    for l in open(H / f"ref/{chain}.pdb"):
        if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[16] in " A":
            res.append((int(l[22:26]), AA.get(l[17:20], "X")))
    best = max(range(-30, 31), key=lambda o: sum(1 for n, a in res if 0 < n + o <= len(useq) and useq[n + o - 1] == a))
    ident = sum(1 for n, a in res if 0 < n + best <= len(useq) and useq[n + best - 1] == a) / len(res)
    idx_of = {n + best: i for i, (n, a) in enumerate(res)}          # UniProt position -> chain residue index
    for r in U[U.lead == lead].itertuples():
        am, mk, ar = r.aln_model, r.aln_marks, r.aln_ref
        col_of_ref, i2, i1, model_of_col = {}, -1, -1, {}
        for c, (x, y) in enumerate(zip(am, ar)):
            if y != "-":
                i2 += 1
                col_of_ref[i2] = c
            if x != "-":
                i1 += 1
        ok = tot = 0
        for t, p, lig in feats:
            i = idx_of.get(p)
            if i is None or i not in col_of_ref:
                rrow.append(dict(lead=lead, protein_id=r.protein_id, role=r.role, feature=t, position=p, ligand=lig, ref=useq[p - 1],
                                 model="(unresolved)", mark=""))
                continue
            c = col_of_ref[i]
            m = am[c]
            inplace = m == useq[p - 1] and mk[c] == ":"
            tot += 1
            ok += inplace
            rrow.append(dict(lead=lead, protein_id=r.protein_id, role=r.role, feature=t, position=p, ligand=lig, ref=useq[p - 1],
                             model=m, mark=mk[c], in_place=inplace))
        srow.append(dict(lead=lead, protein_id=r.protein_id, role=r.role, TM_model=r.TM1, TM_reference=r.TM2, RMSD=r.RMSD, Lali=r.Lali,
                         plddt=pl.get(r.protein_id), residues_in_place=f"{ok}/{tot}" if feats else "n/a",
                         ref_chain=chain, uniprot_offset=best, chain_uniprot_identity=round(ident, 3)))
S = pd.DataFrame(srow)
S.to_csv(H / "data/rung4_summary.tsv", sep="\t", index=False)
pd.DataFrame(rrow).to_csv(H / "data/rung4_residues.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(S.to_string(index=False))
