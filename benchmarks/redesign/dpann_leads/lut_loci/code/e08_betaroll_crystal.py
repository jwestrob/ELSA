#!/usr/bin/env python
"""lut_loci e08 (Mac): the fixed crystal check (tests_rule.json T_crystal_check, s02 center finder unchanged) on the curated
beta-roll calcium-binding control 1KAP (P. aeruginosa serralysin AprA, UniProt Q03023, reviewed; 8 Ca), plus a descriptive
oxygen-cluster readout (not part of the rule): per protein-bound metal, protein O within 3.0 A split into backbone O,
carboxylate O and other side-chain O, and the number of distinct Asp/Glu reaching it (s02 reach rule)."""
import sys

import numpy as np
import pandas as pd

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
sys.path.insert(0, B + "/code")
import s02_sites as S2  # noqa: E402

A = S2.parse(B + "/crystals/1KAP.pdb")
P = A[~A.het]
M = A[A.het & A.resn.isin(["CA", "ZN"])]
C = S2.centers(P, None, None)
xyz = P[["x", "y", "z"]].to_numpy()
rows = []
for m in M.itertuples():
    d = np.linalg.norm(xyz - np.array([m.x, m.y, m.z]), axis=1)
    nb = P[(d <= 3.0) & (P.el == "O")]
    carb = nb.apply(lambda r: r.resn in S2.CARBO and r["name"] in S2.CARBO[r.resn], axis=1) if len(nb) else pd.Series([], dtype=bool)
    dc = np.linalg.norm(C[["x", "y", "z"]].to_numpy() - np.array([m.x, m.y, m.z]), axis=1) if len(C) else np.array([])
    cap = int(C[dc <= 2.0].cap.max()) if len(C) and (dc <= 2.0).any() else 0
    rows.append(dict(metal=m.resn, resi=m.resi, chain=m.chain, protein_O=len(nb), backbone_O=int((nb["name"] == "O").sum()),
                     carboxylate_O=int(carb.sum()), other_sc_O=int(len(nb) - (nb["name"] == "O").sum() - carb.sum()),
                     capacity_within2A=cap, recovered=cap >= S2.PASS_CONS))
D = pd.DataFrame(rows)
D.to_csv(B + "/data/sites_crystal_1KAP.tsv", sep="\t", index=False)
print(D.to_string(index=False))
ca = D[(D.metal == "CA") & (D.protein_O >= 2)]
print("protein-bound Ca recovered:", int(ca.recovered.sum()), "/", len(ca))
