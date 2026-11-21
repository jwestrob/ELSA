#!/usr/bin/env python
"""lut_loci e06 (Mac; s05 with EF follow-up paths): read the LA and CA co-folds (tests_rule.json T_cofold).

Per ion: protein O/N donors within 3.0 A (definition of the lanthanide project's site_cn.py), donor types (carboxylate O,
amide O of Asn/Gln, backbone O, hydroxyl O, other O, N), chelating carboxylates (both O within 3.0 A), distinct Asp/Glu,
metal-O distances, mean pLDDT of donor residues (CA B-factor), overlap of its Asp/Glu with the apo best center (sites_apo.tsv).
Best ion per model = most protein O donors, then most Asp/Glu. PASS_cofold (LA): >= 5 protein O from >= 3 Asp/Glu and >= 3 of
them among the apo best center's reaching residues. Writes data/cofold_ions.tsv and data/cofold_best.tsv.
"""
import sys

import numpy as np
import pandas as pd

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
sys.path.insert(0, B + "/code")
from s02_sites import parse  # noqa: E402

CARBO = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
AMIDE = {"ASN": "OD1", "GLN": "OE1"}
HYDROX = {"SER": "OG", "THR": "OG1", "TYR": "OH"}
apo = pd.read_csv(B + "/data/sites_ef.tsv", sep="\t").set_index("name")
names = [l[1:].strip() for l in open(B + "/folds_in/ef_cofold.faa") if l.startswith(">")]
rows = []
for ion in ["LA", "CA"]:
    for nm in names:
        A = parse(f"{B}/folds/ef_cofold_{ion}/{nm}.pdb")
        P = A[~A.het]
        M = A[A.het & (A.resn.str.upper() == ion)]
        pl = P[P["name"] == "CA"].set_index("resi").b.to_dict()
        xyz = P[["x", "y", "z"]].to_numpy()
        best_res = str(apo.loc[nm, "best_res"]) if isinstance(apo.loc[nm, "best_res"], str) else ""
        apo_set = {int(x[1:]) for x in best_res.split(";") if x}
        for m in M.itertuples():
            d = np.linalg.norm(xyz - np.array([m.x, m.y, m.z]), axis=1)
            nb = P[(d <= 3.0) & P.el.isin(["O", "N"])].copy()
            nb["d"] = d[(d <= 3.0) & P.el.isin(["O", "N"]).to_numpy()]
            typ = []
            for r in nb.itertuples():
                if r.resn in CARBO and r.name in CARBO[r.resn]:
                    typ.append("carboxylate")
                elif r.resn in AMIDE and r.name == AMIDE[r.resn]:
                    typ.append("amide_O")
                elif r.name == "O":
                    typ.append("backbone_O")
                elif r.resn in HYDROX and r.name == HYDROX[r.resn]:
                    typ.append("hydroxyl_O")
                elif r.el == "O":
                    typ.append("other_O")
                else:
                    typ.append("N")
            nb["typ"] = typ
            O = nb[nb.el == "O"]
            acid = O[O.typ == "carboxylate"].resi.unique()
            chel = sum(1 for (ri, rn), g in O[O.typ == "carboxylate"].groupby(["resi", "resn"]) if len(g) == 2)
            rows.append(dict(name=nm, role=nm.split("__")[0], family=nm.split("__")[1], ion=ion, ion_resi=m.resi,
                             protein_O=len(O), protein_ON=len(nb), carboxylate_O=int((O.typ == "carboxylate").sum()),
                             amide_O=int((O.typ == "amide_O").sum()), backbone_O=int((O.typ == "backbone_O").sum()),
                             hydroxyl_O=int((O.typ == "hydroxyl_O").sum()), N=int((nb.typ == "N").sum()),
                             acid_res=len(acid), chelating=chel,
                             mO_mean=round(O.d.mean(), 2) if len(O) else np.nan, mO_min=round(O.d.min(), 2) if len(O) else np.nan,
                             mO_max=round(O.d.max(), 2) if len(O) else np.nan,
                             donor_plddt=round(float(np.mean([pl.get(r, np.nan) for r in nb.resi.unique()])), 1) if len(nb) else np.nan,
                             overlap_apo=len(set(int(x) for x in acid) & apo_set),
                             acid_list=";".join(f"{O[O.resi == r].resn.iloc[0][0]}{r}" for r in acid)))
I = pd.DataFrame(rows)
I.to_csv(B + "/data/ef_cofold_ions.tsv", sep="\t", index=False)
Bst = I.sort_values(["protein_O", "acid_res"], ascending=False).drop_duplicates(["name", "ion"])
Bst["PASS_cofold_rule"] = (Bst.protein_O >= 5) & (Bst.acid_res >= 3) & (Bst.overlap_apo >= 3)
Bst.to_csv(B + "/data/ef_cofold_best.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
cols = ["name", "ion", "protein_O", "carboxylate_O", "amide_O", "backbone_O", "hydroxyl_O", "acid_res", "chelating", "mO_mean",
        "donor_plddt", "overlap_apo", "PASS_cofold_rule"]
print(Bst.sort_values(["role", "name", "ion"])[cols].to_string(index=False))
la = Bst[Bst.ion == "LA"].set_index("name")
ca = Bst[Bst.ion == "CA"].set_index("name")
same = (la.protein_O == ca.protein_O.reindex(la.index)) & (la.acid_res == ca.acid_res.reindex(la.index))
print("LA vs CA identical donor counts:", int(same.sum()), "/", len(la))
print(Bst[Bst.ion == "LA"].groupby("role").agg(n=("name", "size"), pass_rule=("PASS_cofold_rule", "sum"),
                                                O_ge5_acid_ge3=("protein_O", lambda s: 0)).to_string())
