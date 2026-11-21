#!/usr/bin/env python
"""ribo_audit part 3, supplementary tables: normalised order variability (rearr3.py), alpha-block placement (alpha3.py,
POST HOC), shares of the bacterial shortfall (predeclared3 mapping), extension contents, cluster-level definition (POST HOC).
Usage: tables3b.py WORK3DIR OUTFILE  (WORK3DIR holds the work3 files and loose/)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(sys.argv[1])
out = []
pr = out.append
GT = ("Bact_closed", "Bact_multi", "Archaea")

# ---------------- shares
pr("SHARES OF THE BACTERIAL SHORTFALL (predeclared3 mapping; denominators = unresolved bacterial runs)")
for var in ("strict", "loose"):
    C = pd.read_csv(W / ("cases3.tsv" if var == "strict" else "loose/cases3.tsv"), sep="\t")
    for o in ("PRIMARY", "SECONDARY"):
        for gts in (("Bact_closed",), ("Bact_multi",), ("Bact_closed", "Bact_multi")):
            c = C[(C.output == o) & C.gtype.isin(gts)]
            n = len(c)
            v = c.verdict.value_counts()
            st = c.subtype.fillna("")
            rearr = 0  # verdict 1 at a rare pair; there is no verdict 1 in Bacteria
            over = int(((c.verdict == 4) & st.str.startswith("4b")).sum())
            other = int(((c.verdict == 4) & ~st.str.startswith("4b")).sum()) + int(v.get(1, 0))
            sc = int(v.get(5, 0))
            rd = int(v.get(2, 0) + v.get(3, 0))
            dat = int(v.get(6, 0) + v.get(7, 0))
            lab = "+".join(g.replace("Bact_", "") for g in gts)
            pr(f"  {var:6s} {o:9s} {lab:12s} n={n:3d}  rearrangement {rearr} ({rearr / n:.1%})  over-extension into "
               f"conserved neighbours {over} ({over / n:.1%})  scorer {sc} ({sc / n:.1%})  run definition {rd} "
               f"({rd / n:.1%}; v2 {int(v.get(2, 0))}, v3 {int(v.get(3, 0))})  other method {other}  data {dat}")

# ---------------- order variability, normalised names
pr("")
pr("ORDER VARIABILITY, ribosomal-protein names normalised (rearr3.py; archaeal Pfam synonyms mapped, POST HOC naming fix)")
P = pd.read_parquet(W / "pair_shares_norm.parquet")
R = pd.read_parquet(W / "run_order_norm.parquet")
S1 = pd.read_parquet(W / "std3_PRIMARY.parquet")[["run_id", "resolved"]]
S2 = pd.read_parquet(W / "std3_SECONDARY.parquet")[["run_id", "resolved"]].rename(columns={"resolved": "res2"})
R = R.merge(S1, on="run_id").merge(S2, on="run_id")
alpha = {"S13", "S36", "L36"}
for gt in GT:
    p = P[P.gtype == gt]
    x = R[R.gtype == gt]
    rare = p[p.share < 0.10]
    spc_alpha = rare.apply(lambda r: {r.a, r.b} in ({"L15", "L36"}, {"L15", "S13"}), axis=1).sum() if len(rare) else 0
    selfp = int((rare.a == rare.b).sum())
    pr(f"  {gt:11s} runs {len(x)}  marker pairs {len(p)}  other groups per pair median {p.n_groups.median():.0f}  share median "
       f"{p.share.median():.3f}  q25 {p.share.quantile(.25):.3f}  q10 {p.share.quantile(.1):.3f}  run-minimum median "
       f"{x.min_share.median():.3f}")
    pr(f"              rare pairs (< 0.10) {len(rare)} ({len(rare) / len(p):.2%}) in {int((x.n_rare >= 1).sum())} runs; of "
       f"these L15|L36 or L15|S13 {spc_alpha}, same-name pairs (split / duplicate calls) {selfp}, other {len(rare) - spc_alpha - selfp}")
    pr(f"              runs with >= 1 rare pair: PRIMARY resolved {int(((x.n_rare >= 1) & x.resolved).sum())}/{int(x.resolved.sum())}, "
       f"unresolved {int(((x.n_rare >= 1) & ~x.resolved).sum())}/{int((~x.resolved).sum())}; SECONDARY resolved "
       f"{int(((x.n_rare >= 1) & x.res2).sum())}/{int(x.res2.sum())}, unresolved {int(((x.n_rare >= 1) & ~x.res2).sum())}/{int((~x.res2).sum())}")
    if gt == "Archaea":
        pr("              top rare pairs: " + str(rare.groupby(["a", "b"]).size().sort_values(ascending=False).head(8).to_dict()))
canon = "L15-S5-L18-L6-S8-S14-L5-L24-L14-S17-L29-S3-L22-S19-L2-L23-L4-L3".split("-")
pos = {c: i for i, c in enumerate(canon)}


def colinear(o):
    s = [pos[c] for c in o.split("-") if c]
    return all(a < b for a, b in zip(s, s[1:])) or all(a > b for a, b in zip(s, s[1:]))


for d, gts in (("Bacteria", ("Bact_closed", "Bact_multi")), ("Archaea", ("Archaea",))):
    x = R[R.gtype.isin(gts)]
    full = x[x.n_core == 18]
    col = x.core_order.map(colinear)
    pr(f"  {d}: core-marker orders distinct {x.core_order.nunique()} over {len(x)} runs ({x.group.nunique()} groups); modal "
       f"share {x.core_order.value_counts().iloc[0] / len(x):.3f}; runs with all 18 core markers {len(full)} in "
       f"{full.core_order.nunique()} order(s); runs collinear with the one bacterial S10-spc order {int(col.sum())}/{len(x)}; "
       f"core markers per run {x.n_core.value_counts().sort_index().to_dict()}")
    for o, k in x[~col].core_order.value_counts().items():
        pr(f"      non-collinear: {o} x{k}")

# ---------------- alpha block placement (POST HOC)
pr("")
pr("WHERE THE ALPHA BLOCK (L36 / S13 / S11 / S4 / L17) SITS RELATIVE TO EACH BACTERIAL RUN (alpha3.py, POST HOC, output-independent)")
A = pd.read_parquet(W / "alpha3.parquet")
E3 = pd.read_parquet(W / "std3_ENGINE.parquet")[["run_id", "resolved"]].rename(columns={"resolved": "res3"})
A = A.merge(S1, on="run_id").merge(S2, on="run_id").merge(E3, on="run_id")
A["edge"] = np.minimum(A.to_contig_start, A.to_contig_end)


def cat(r):
    if r.alpha_in_run:
        return "a alpha block inside the run (spc-alpha insert <= 2 genes)"
    if r.alpha_next12:
        return "b alpha block <= 12 genes past the run, same contig"
    if r.edge <= 12:
        return "c run <= 12 genes from a contig end, alpha block absent"
    return "d alpha block absent within 12 genes, contig continues"


A["cat"] = A.apply(cat, axis=1)
t = A.groupby(["gtype", "cat"]).agg(runs=("run_id", "size"), PRIMARY=("resolved", "sum"), SECONDARY=("res2", "sum"),
                                     ENGINE=("res3", "sum"))
t["PRIMARY_unresolved"] = t.runs - t.PRIMARY
pr(t.to_string())
pr("  inserts between L15 and the alpha block in category a: " + str(A[A.between_L15_alpha != ""].between_L15_alpha.value_counts().to_dict()))
A["near_edge"] = A.edge <= 12
t2 = A.groupby(["gtype", "near_edge"]).agg(runs=("run_id", "size"), PRIMARY=("resolved", "sum"), SECONDARY=("res2", "sum"),
                                            ENGINE=("res3", "sum"))
pr("  by distance to a contig end (<= 12 genes):")
pr(t2.to_string())
C = pd.read_csv(W / "cases3.tsv", sep="\t")
m = C[C.output == "PRIMARY"].merge(A[["run_id", "cat"]], on="run_id")
pr("  PRIMARY strict verdicts by category:")
pr(pd.crosstab([m.gtype, m.cat.str[0]], m.verdict).to_string())

# ---------------- extension contents
pr("")
pr("GENES IN THE OUTPUT'S BEST LOCUS BEYOND THE RUN (unresolved runs with verdicts 3-5; ext3.parquet, strict)")
X = pd.read_parquet(W / "ext3.parquet").rename(columns={"where": "wh"})
for o in ("PRIMARY", "SECONDARY"):
    for gt in GT:
        e = X[(X.output == o) & (X.gtype == gt)]
        if not len(e):
            continue
        for w in ("C_beyond_run", "best_beyond_C"):
            q = e[e.wh == w]
            if not len(q):
                continue
            lab = np.where(q.cls.isin(["R", "N"]), q.lit.fillna("").replace({"rpoD": "rpoA/rpoD"}) + "[" + q.cls + "]",
                           q.top.fillna("unannotated"))
            pr(f"  {o:9s} {gt:11s} {w:13s} cases {q.case_key.nunique():3d} genes {len(q):4d}: ribosomal {int((q.cls == 'R').sum())}, "
               f"literature members {int((q.cls == 'N').sum())}, other {int((q.cls == 'X').sum())} (unannotated "
               f"{int(q.unannot.fillna(False).sum())})")
            pr("      " + str(pd.Series(lab).value_counts().head(20).to_dict()))

# ---------------- cluster-level definition (POST HOC)
pr("")
pr("CLUSTER-LEVEL RUN DEFINITION (POST HOC: markers = ribosomal proteins + literature cluster members, same <= 2-gene gap rule; "
   "standard 80 / 80 best locus; chance = 5 random windows of the extended-run length per target, seed 17)")
J = json.load(open(W / "extdef.json"))
for o in ("PRIMARY", "SECONDARY", "ENGINE"):
    b = sum(J[o][g]["resolved"] for g in ("Bact_closed", "Bact_multi"))
    pr(f"  {o:9s} Bacteria {b}/658 (closed {J[o]['Bact_closed']['resolved']}/260, multi {J[o]['Bact_multi']['resolved']}/398)  "
       f"Archaea {J[o]['Archaea']['resolved']}/145  chance closed {J[o]['Bact_closed']['chance']:.2%} multi "
       f"{J[o]['Bact_multi']['chance']:.2%} Archaea {J[o]['Archaea']['chance']:.2%}")
pr("  extended run vs run (median genes): " + json.dumps(J["extension"]))

Path(sys.argv[2]).write_text("\n".join(out) + "\n")
print("\n".join(out))
