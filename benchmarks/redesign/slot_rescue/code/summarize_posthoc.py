#!/usr/bin/env python3
"""POST HOC table (not predeclared): whole-list votes (full, fullg) and the slot-consistent whole list (combfull) beside the
predeclared slot call, same sets and controls as summarize.py. Writes results/posthoc_table.tsv."""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parents[1]
R, D = HERE / "results", HERE / "data"
w = pd.read_csv(R / "per_gene.tsv.gz", sep="\t", keep_default_na=False, low_memory=False).set_index("row_id")
ph = pd.read_parquet(R / "posthoc.parquet")
L = pd.read_csv(D / "labels.tsv", sep="\t", keep_default_na=False)
U = pd.read_csv(D / "units.tsv", sep="\t", keep_default_na=False)
pred = {p: set(L.label[L[p] == 1]) for p in ["ribosomal_protein", "sec_pathway", "rna_polymerase", "atp_synthesis", "trna_modification", "membrane", "cytochrome_bd_oxidase"]}
CORE = set(pred["ribosomal_protein"])
for r in U[U.set.isin(["lead_X", "DL_slot", "DL_profile"])].itertuples():
    CORE |= {x for x in (r.R_pfam.split(";") + r.R_ko.split(";")) if x}
for nm in ("full", "fullg", "combfull"):
    for arm in ("Z", "A"):
        x = ph[ph.arm == arm].set_index("row_id")
        w[f"{nm}_{arm}_raw"] = x[f"{nm}_class"]; w[f"{nm}_{arm}_label"] = x[f"{nm}_label"]
        w[f"{nm}_{arm}"] = ["right" if a == "right" else ("broad" if a == "other" and b and l in pred.get(b, set()) else ("other" if a == "other" else "none"))
                            for a, l, b in zip(w[f"{nm}_{arm}_raw"], w[f"{nm}_{arm}_label"], w.broad_class)]
routes = ["list_Z", "list50_Z", "full_Z", "fullg_Z", "combfull_Z", "slot_Z", "list_A", "list50_A", "full_A", "fullg_A", "combfull_A", "slot_A"]
tr, ct = w[w.role == "truth"], w[w.role == "control"]
sets = {"lead_X verified": tr[(tr.set == "lead_X") & (tr.status == "verified")], "lead_X UNVERIFIED": tr[(tr.set == "lead_X") & (tr.status == "UNVERIFIED")],
        "DL_slot": tr[tr.set == "DL_slot"], "DL_slot eL39": tr[(tr.set == "DL_slot") & (tr.unit == "eL39")], "DL_profile": tr[tr.set == "DL_profile"],
        "DL_profile eL39": tr[(tr.set == "DL_profile") & (tr.unit == "eL39")]}
rows = []
for nm, df in sets.items():
    df = df.drop_duplicates("protein_id")
    rows.append(dict(set=nm, n=len(df), **{r: f"{(df[r] == 'right').sum()} ({(df[r] == 'right').mean():.1%})" for r in routes}))
ver_ids = set(sets["lead_X verified"].protein_id)
for nm, cc in {"controls (verified-matched): false rescue": ct[ct.matched_truth.isin(ver_ids)], "controls (DL-matched): false rescue": ct[ct.set.str.startswith("control_DL")]}.items():
    rec = dict(set=nm, n=len(cc))
    for r in routes:
        f = (cc[r] == "right") | ((cc[f"{r}_raw"] == "other") & cc[f"{r}_label"].isin(CORE))
        rec[r] = f"{int(f.sum())} ({f.mean():.2%})"
    rows.append(rec)
ng = tr[tr.set == "lead_negative"]
rows.append(dict(set="negatives: named the lead's X", n=len(ng), **{r: f"{int((ng[r] == 'right').sum())}" for r in routes}))
out = pd.DataFrame(rows)
out.to_csv(R / "posthoc_table.tsv", sep="\t", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
print(out.to_string())
v = sets["lead_X verified"].drop_duplicates("protein_id")
print("verified: slot_Z right & fullg_Z right", int(((v.slot_Z == "right") & (v.fullg_Z == "right")).sum()), "slot only", int(((v.slot_Z == "right") & (v.fullg_Z != "right")).sum()), "fullg only", int(((v.slot_Z != "right") & (v.fullg_Z == "right")).sum()))
print("verified: slot_A right & fullg_A right", int(((v.slot_A == "right") & (v.fullg_A == "right")).sum()), "slot only", int(((v.slot_A == "right") & (v.fullg_A != "right")).sum()), "fullg only", int(((v.slot_A != "right") & (v.fullg_A == "right")).sum()))
