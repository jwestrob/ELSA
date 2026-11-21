#!/usr/bin/env python
"""Step 10: verdict and one-line reason per passing gene in leads.tsv; brief columns first.
Verdict vocabulary: SUPPORTED (assay pass, unannotated, function otherwise absent from the genome), SUPPORTED-housekeeping,
SUPPORTED-redundant (the genome already carries the function by another source), PASS-NAMED (passes, but a CDD / PDB text
source already names the function: outside the profile-invisible, unannotated population)."""
from pathlib import Path
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/invisible")
L = pd.read_csv(H / "leads.tsv", sep="\t", keep_default_na=False)
WHY = {"K00965": "GalT-branch HIT site (H186 equivalent) and UDP-glucose residues in place; glycogen-locus neighbors; galK absent clade-wide; substrate UNVERIFIED",
       "K01448": "Amidase_3 Zn site (His, Glu, His) and catalytic Glu in place; co-fold coordinates Zn as the positive; substrate UNVERIFIED",
       "K02683": "PriS catalytic Asp pair in place; primase small subunit (heterodimer with PriL); routine housekeeping",
       "K00560": "ThyA catalytic Cys and dUMP / folate residues in place; dNTP locus; routine housekeeping",
       "K00948": "PRPP synthase Lys and ATP / R5P residues in place; genome carries an annotated PRPP synthase",
       "K01299": "M32 carboxypeptidase Glu and Co ligands in place; genome carries an annotated M32"}
v, why = [], []
for r in L.itertuples():
    f = r.function.split()[0]
    if r.named_by_amendment in (True, "True"):
        v.append("PASS-NAMED"); why.append("passes; " + r.amendment_detail.split(";")[0] + " already names the function")
        continue
    if f in ("K02683", "K00560"):
        v.append("SUPPORTED-housekeeping")
    elif str(r.function_absent_otherwise) in ("False", "0"):
        v.append("SUPPORTED-redundant")
    else:
        v.append("SUPPORTED")
    why.append(WHY.get(f, ""))
L["verdict"], L["one_line_reason"] = v, why
first = ["lead", "family_label", "clade", "genes", "genomes", "clade_genomes", "median_length", "tier_T1", "support_median",
         "catalytic_residues_ok", "structure_TM_to_reference", "function_absent_otherwise", "embedding_only_share", "verdict", "one_line_reason"]
L = L[first + [c for c in L.columns if c not in first]]
rank = {"SUPPORTED": 0, "SUPPORTED-housekeeping": 1, "SUPPORTED-redundant": 2, "PASS-NAMED": 3}
L = L.assign(_r=L.verdict.map(rank)).sort_values(["_r", "function", "structure_TM_to_reference"], ascending=[True, True, False]).drop(columns="_r")
L.to_csv(H / "leads.tsv", sep="\t", index=False)
print(L.groupby(["verdict", L.function.str[:6]]).size())
