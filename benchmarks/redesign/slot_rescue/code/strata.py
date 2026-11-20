#!/usr/bin/env python3
"""Stratification (not predeclared): verified lead_X proteins that carry no right-family KO themselves (no Pfam, no R KO)
versus those that carry one. Same routes, genome bootstrap (2,000, seed 17), matched controls. Writes results/strata.tsv."""
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parents[1]
R, D = HERE / "results", HERE / "data"
w = pd.read_csv(R / "per_gene.tsv.gz", sep="\t", keep_default_na=False, low_memory=False).set_index("row_id")
ph = pd.read_parquet(R / "posthoc.parquet")
T = pd.read_csv(D / "truth.tsv", sep="\t", keep_default_na=False)
L = pd.read_csv(D / "labels.tsv", sep="\t", keep_default_na=False)
U = pd.read_csv(D / "units.tsv", sep="\t", keep_default_na=False)
CORE = set(L.label[L.ribosomal_protein == 1])
for r in U[U.set.isin(["lead_X", "DL_slot", "DL_profile"])].itertuples():
    CORE |= {x for x in (r.R_pfam.split(";") + r.R_ko.split(";")) if x}
for nm in ("full", "combfull"):
    for arm in ("Z", "A"):
        x = ph[ph.arm == arm].set_index("row_id")
        w[f"{nm}_{arm}_raw"] = x[f"{nm}_class"]; w[f"{nm}_{arm}_label"] = x[f"{nm}_label"]
        w[f"{nm}_{arm}"] = np.where(x[f"{nm}_class"] == "right", "right", "x")
ko = T.groupby("protein_id").carries_R_ko.max()
routes = ["list_Z", "list_A", "slot_Z", "slot_A", "comb_Z", "comb_A", "full_Z", "full_A", "combfull_Z", "combfull_A"]
v = w[(w.role == "truth") & (w.set == "lead_X") & (w.status == "verified")].drop_duplicates("protein_id").copy()
v["rko"] = v.protein_id.map(ko)
c = w[w.role == "control"]
rng = np.random.default_rng(17)
rows = []
for lab, sub in (("no Pfam, no right-family KO", v[v.rko == 0]), ("no Pfam, carries the right-family KO", v[v.rko == 1])):
    ub, inv = np.unique(sub.bin_id, return_inverse=True)
    draws = rng.integers(0, len(ub), size=(2000, len(ub)))
    cnt = np.bincount(inv, minlength=len(ub)).astype(float)
    rec = dict(stratum=lab, n=len(sub))
    cc = c[c.matched_truth.isin(set(sub.protein_id))]
    rec["n_controls"] = len(cc)
    for r in routes:
        x = (sub[r] == "right").to_numpy(float)
        S = np.bincount(inv, weights=x, minlength=len(ub))
        est = np.array([S[d].sum() / cnt[d].sum() for d in draws])
        lo, hi = np.quantile(est, [.025, .975])
        rec[r] = f"{int(x.sum())} ({x.mean():.1%} [{lo:.1%}, {hi:.1%}])"
        f = (cc[r] == "right") | ((cc[f"{r}_raw"] == "other") & cc[f"{r}_label"].isin(CORE))
        rec[r + "_ctrl"] = f"{int(f.sum())} ({f.mean():.2%})"
    rows.append(rec)
out = pd.DataFrame(rows)
out.to_csv(R / "strata.tsv", sep="\t", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(out.T.to_string())
